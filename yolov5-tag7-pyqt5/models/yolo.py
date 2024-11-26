# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
YOLO-specific modules

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse # 解析命令行参数模块
import contextlib
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path # Path将str转换为Path对象，使字符串路径易于操作

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != 'Windows':
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import *
from models.experimental import *
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (fuse_conv_and_bn, initialize_weights, model_info, profile, scale_img, select_device,
                               time_sync)
# 导入thop包，用于计算FLOPs
try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    """
    Detect模块是用来构建Detect层的，将输入的feature map通过一个1x1卷积和公式计算到我们想要的shape，为后面的计算损失或者NMS后处理作准备
    """
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        '''
        :params nc: number of classes
        :params anchors: 传入3个预测特征层上的所有anchor的大小(P3/P4/P5)
        :params ch: [256,512,1024]*width_multiple->[128,256,512]，即3个预测特征层向Detect输入的channel，也是前3个C3层输出的channel
        '''
        super().__init__()
        # nc:分类数量
        self.nc = nc  # number of classes
        # no:每个anchor的输出数
        self.no = nc + 5  # number of outputs per anchor
        # nl:预测层数(3)，也是Detect层的个数
        self.nl = len(anchors)  # number of detection layers
        # na:anchors的数量(3)
        self.na = len(anchors[0]) // 2  # number of anchors
        # grid:格子坐标系
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid；[tensor([]),tensor([]),tensor([])]
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid；[tensor([]),tensor([]),tensor([])]
        '''
        模型中需要保存的参数一般有两种：
        一种是反向传播需要被optimizer更新的，称为parameter;另一种不需要被更新，称为buffer
        buffer的参数更新是在forward中，而optim.step只能更新nn.parameter参数
        '''
        # 写入缓存中，并命名为anchors
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        # 每个预测特征层向Detect传入的feature map都要调用一次1x1卷积，卷积到self.no*self.na的通道，达到全连接的作用
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        # 一般是True，默认不使用AWS，Inferentia加速
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x): # x是三个预测特征层向Detect输入的feature map
        '''
        如果是训练阶段，返回通过1x1卷积层并转换通道后的x，shape为[bs,na,ny,nx,no]；
        如果是推理阶段，对于Detect模块（Segment模块有些许差异），先由通过1x1卷积层并转换通道后的x切分为xy,wh,confidence，通过公式计算得到预测的xy,wh，再与confidence拼接起来然后融合通道，最后输出，输出的shape为[bs,na*nx*ny,no]
        '''
        z = []  # inference output
        for i in range(self.nl):
            # x.shape为[bs,128/256/512,ny,nx]，bs为batch_size，下面以[bs,128,ny,nx]为例
            x[i] = self.m[i](x[i])  # conv；x.shape from [bs,128,32,32] to [bs,255(na*no),32,32]
            bs, _, ny, nx = x[i].shape
            # x(bs,255,ny,nx) to x(bs,3,85,ny,nx) to x(bs,3,ny,nx,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            '''
            因为推理返回的不是归一化后的网络偏移量，需要加上网格的位置，得到最终的推理坐标，再送入NMS
            所以这里构建网格就是为了记录每个grid的网格坐标，方便后面使用
            '''
            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    '''
                    此时
                    xy*2代表对应网格左上角坐标c_x,c_y
                    self.grid[i]代表坐标偏移量2*\sigma(t_x)-0.5,2*\sigma(t_y)-0.5
                    self.anchor_grid[i]代表p_w,p_h
                    wh代表\sigma(t_w),\sigma(t_h)
                    '''
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy；根据公式计算预测目标中心坐标b_x,b_y，b_x=(2*\sigma(t_x)-0.5)+c_x,b_y=(2*\sigma(t_y)-0.5)+c_y，并还原尺度
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh；根据公式计算预测目标宽高b_w,b_h，b_w=p_w*(2*\sigma(t_w))^2,b_h=p_h*(2*\sigma(t_h))^2
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    # 相对坐标转换到grid绝对坐标系
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, '1.10.0')):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing='ij') if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5；根据公式计算预测目标中心相对坐标2*\sigma(t_x)-0.5,2*\sigma(t_y)-0.5，注意还未加c_x,c_y
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    # YOLOv5 base model
    def forward(self, x, profile=False, visualize=False):
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # 前向传播具体实现
    def _forward_once(self, x, profile=False, visualize=False):
        """
        @params x: 输入图像
        @params profile: True可以做一些性能评估
        @params visualize: True可以做一些特征可视化
        @return x：输入图像依次经过所有层结构后输出的feature map
        """
        # y: 存放着self.save=True的每一层的输出，因为后面层结构Concat操作要用到
        # dt: 在profile中做性能评估时使用
        y, dt = [], []  # outputs
        # 前向推理每一层结构。m.i=index，m.f=from，m.type=类名，m.np=number of params
        for m in self.model:
            if m.f != -1:  # if not from previous layer；m.f=当前层的输入来自哪一层的输出；只有Concat层和Detect层的m.f不是-1
                # 所以这里一共只需要做4个Concat操作和一个Detect操作，其他层操作跳过这步
                # Concat: 如m.f=[-1,6] x就有两个元素，一个是上一层的输出，一个是index=6的层的输出，再送到x=m(x)做Concat操作
                # Detect: 如m.f=[17, 20, 23] x就有三个元素，分别存放第17层第20层第23层的输出，再送到x=m(x)做Detect的forward
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            # 打印日志信息：FLOPs time等
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run；正向推理
            y.append(x if m.i in self.save else None)  # save output；存放self.save的每一层的输出，因为后面需要用来做Concat等操作，不在self.save的层的输出就设为None
            # 特征可视化，可以自己改动想要那层的特征进行可视化
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    # 打印日志信息，前向推理时间
    def _profile_one_layer(self, m, x, dt):
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1E9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f'{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}')
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    # fuse()是用来进行conv和bn层合并，为了加快模型推理速度
    def fuse(self):  # fuse model Conv2d() + BatchNorm2d() layers
        """用在detect.py、val.py
        fuse model Conv2d() + BatchNorm2d() layers
        调用torch_utils.py中的fuse_conv_and_bn函数和common.py中Conv模块的fuseforward函数
        """
        LOGGER.info('Fusing layers... ')
        for m in self.model.modules():
            # 如果当前层是卷积层Conv且有bn结构，那么就调用fuse_conv_and_bn函数将conv和bn进行融合，加速推理
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward；更新前向传播(反向传播不用管，因为这种推理只用在推理阶段)
        self.info() # 打印conv+bn融合后的模型信息
        return self

    # 打印模型结构信息，在当前类__init__函数结尾处有调用
    def info(self, verbose=False, img_size=640):  # print model information
        model_info(self, verbose, img_size)

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg='yolov5s.yaml', ch=3, nc=None, anchors=None):  # model, input channels, number of classes
        """
        :params cfg: 模型配置文件
        :params ch: input img channels 一般是 3（RGB文件）
        :params nc: number of classes 数据集的类别个数
        :anchors: 一般是 None
        """
        super().__init__()
        # 如果cfg已经是字典，则直接赋值，否则先加载cfg路径的文件为字典并赋值给self.yaml
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub
            self.yaml_file = Path(cfg).name
            # 如果配置文件中有中文，打开时要加encoding参数
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)  # model dict；从yaml文件中加载出字典

        # Define model
        # ch:输入通道数。假如self.yaml有键‘ch’，则将该键对应的值赋给变量ch；假如没有‘ch’，则将形参ch赋给变量ch，并新创建一个键：self.yaml['ch']，将形参ch也赋给这个键
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)  # input channels；self.yaml.get('ch',ch)这里如果yaml字典中找不到键’ch‘，就返回默认值ch(3)了
        # 假如yaml中的nc和传入形参中的nc不一致（前提nc!=None），则覆盖yaml中的nc，一般跳过这步
        if nc and nc != self.yaml['nc']:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml['nc'] = nc  # override yaml value
        # 前提anchors!=None，则用传入形参中的anchors覆盖yaml中的anchors，一般跳过这步
        if anchors:
            LOGGER.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)  # override yaml value
        # 得到模型，以及index保存标签
        # self.model: 初始化的整个网络模型(包括Detect层结构)
        # self.save: 所有层结构中的from不等于-1的序号，并排好序[4,6,10,14,17,20,23]
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml['nc'])]  # default names；初始化类名列表，默认为[0,1,2...]
        # self.inplace=True，默认True，不使用加速推理
        self.inplace = self.yaml.get('inplace', True)

        # Build strides, anchors；获取Detect模块的stride(相对输入图像的下采样率)和anchors在当前Detect输出的feature map的尺寸
        m = self.model[-1]  # Detect()；网络架构的最后一层是Detect层
        if isinstance(m, (Detect, Segment)): # 检验模型的最后一层是Detect模块或Segment模块
            # 下面步骤是用来确定原图到三个预测特征层在Detect模块输出的feature map的下采样倍率stride
            # 但由于无法直接推测得到stride，所以创建一个torch.zeros(1, ch, s, s)大小的“测试图片”进行正向传播，最后得到三个预测特征层
            # 然后将“测试图片”的尺寸[256x256]除以这三个输出的feature map的尺寸[32x32,16x16,8x8]，就得到对应的stride[8x8,16x16,32x32]
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x) # 设置一个lambda函数用来区分Detect模块和Segment模块的正向传播的输出
            # 计算三个下采样的倍率，即[8,16,32]
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            # 检查anchor顺序与stride顺序是否一致，anchor的顺序应该是从小到大，这里排一下序
            check_anchor_order(m)
            # 对应的anchor进行缩放操作，从而得到anchor在实际的特征图中的位置，因为加载的原始anchor大小是相对于原图的像素，但是经过卷积池化之后，特征图的高宽变小了
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once 初始化偏置

        # Init weights, biases
        # 调用torch_utils.py下initialize_weights初始化模型权重
        initialize_weights(self)
        self.info() # 打印模型信息
        LOGGER.info('')

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment: # 是否在测试时也使用数据增强  Test Time Augmentation(TTA)
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # 带数据增强的前向传播
    def _forward_augment(self, x):
        ''' TTA Test Time Augmentation '''
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud上下, 3-lr左右)
        y = []  # outputs
        for si, fi in zip(s, f):
            # scale_img缩放图片尺寸，调用torch_utils.py下的scale_img()函数
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size) # _descale_pred将推理结果恢复到相对原图图片尺寸
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    # 将推理结果恢复到原图图片尺寸(逆操作)
    def _descale_pred(self, p, flips, scale, img_size):
        # de-scale predictions following augmented inference (inverse operation)
        """
        用在上面的_forward_augment函数上
        将推理结果恢复到原图图片尺寸  Test Time Augmentation(TTA)中用到
        :params p: 推理结果
        :params flips: 翻转标记(2-ud上下, 3-lr左右)
        :params scale: 图片缩放比例
        :params img_size: 原图图片尺寸
        """
        # 不同的方式前向推理使用公式不同，具体可看Detect函数
        if self.inplace: # 默认True，不使用AWS Inferentia
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    # 这个是TTA的时候对原图片进行裁剪，也是一种数据增强方式，用在TTA测试的时候
    def _clip_augmented(self, y):
        # Clip YOLOv5 augmented inference tails
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4 ** x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4 ** x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility；使得命名上Model和DetectionModel等效


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg='yolov5s-seg.yaml', ch=3, nc=None, anchors=None):
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):  # yaml, model, number of classes, cutoff index
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        # Create a YOLOv5 classification model from a YOLOv5 detection model
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        # Create a YOLOv5 classification model from a *.yaml file
        self.model = None


def parse_model(d, ch):  # model_dict, input_channels(3)；d是一个字典，ch是一个list，会记录下不同层(yaml中的module)输出的通道数
    """
    用在下面Model模块中
    解析模型文件(字典形式)，并搭建网络结构
    这个函数其实主要做的就是: 更新当前层的args（参数）,计算c2（当前层的输出channel）
                         => 使用当前层的参数搭建当前层
                         => 生成 layers + save
    @Params d: model_dict 模型文件 字典形式 {dict:7} [yolov5s.yaml]中的6个元素+ch
    @Params ch: 记录模型每一层的输出channel 初始ch=[3] 后面会删除
    @return nn.Sequential(*layers): 网络的每一层的层结构
    @return sorted(save): 把所有层结构中from不是-1的值记下 并排序[4, 6, 10, 14, 17, 20, 23]
    """
    # Parse a YOLOv5 model.yaml dictionary
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # 读取d字典中的anchors和parameters(nc、depth_multiple、width_multiple)等
    anchors, nc, gd, gw, act = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple'], d.get('activation') # 从字典中获取参数
    if act: # 一般跳过这步
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors；anchors[0]如[10,13, 16,30, 33,23]，anchors[0]//2得到当前尺度anchors的数量(3)
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)；每个预测特征层的输出channel(255)

    # 开始搭建网络
    # layers: 保存每一层的层结构
    # save: 记录下所有层结构中from中不是-1的层结构序号
    # c2: 保存当前层的输出channel
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # d['backbone']+d['head']是把字典的两个元素的值连接到一起构成一个大的list
    for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):  # from, number, module, args
        m = eval(m) if isinstance(m, str) else m  # eval strings；m本来是一个字符串类型，经过eval(m)后变成了class类型
        for j, a in enumerate(args):
            # args是一个列表，这一步把列表中的内容取出来
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings；head模块中有一些args存在字符串表示的内容，也要转化成对应的参数
        # 将深度与深度因子相乘，计算层深度；深度最小为1
        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain；使用gd控制模型的深度，即对于当前模块决定其重复次数
        # 如果当前的模块m在本项目定义的模块类型中，就可以处理这个模块
        if m in {
                Conv, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, MixConv2d, Focus, CrossConv,
                BottleneckCSP, C3, C3TR, C3SPP, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x}:
            c1, c2 = ch[f], args[0] # ch是一个list，会记录下不同层(yaml中的m)输出的通道数；c1是ch[from]，表示当前模块输入的通道数；c2是yaml中的args[0]，表示当前模块输出的通道数
            if c2 != no:  # if not output
                # make_divisible的作用：使得原始的通道数乘以宽度因子之后取整到8的倍数，这样处理一般是让模型的并行性和推理性能更好
                c2 = make_divisible(c2 * gw, 8) # 使用gw控制模型的宽度，也就是每层的输出通道数；在utils/general.py中 math.ceil(c2*gw/8)*8 向上取整，保证输出通道数是8的倍数（适合gpu并行运算）
            # 将前面的运算结果保存在args中，它也就是这个模块最终的输入参数
            args = [c1, c2, *args[1:]] # *args[1:]为每个模块的 k,s 等,*list表示把list解开为多个独立的元素
            # 根据每层网络参数的不同，分别处理参数
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                # 这里的意思就是重复n次，比如conv这个模块重复n次，这个n是上面算出来的depth
                args.insert(2, n)  # number of repeats；从位置2处插入n，n为模块的个数，会在模块中使用for循环重复多次
                n = 1 # 然后置n=1
        elif m is nn.BatchNorm2d: # yaml中不存在BN module，因此不会执行
            args = [ch[f]]
        elif m is Concat: # 如果是Concat模块，会把输入[f]通道加起来作为输出通道个数
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, 8)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f] # yaml中的 nn.Upsample，输出通道数和上一层的输出通道数一样
        # 构建整个网络模块，这里就是根据模块的重复次数n以及模块本身和它的参数来构建这个模块和参数对应的Module
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module；n表示当前模块的个数也就是循环几次；m(*args)表示实例化一个m对象
        # 打印输出参数个数
        # 获取模块(module type)具体名例如 models.common.Conv, models.common.C3, models.common.SPPF 等
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f'{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}')  # print
        """
        如果x不是-1，则将其保存在save列表中，表示该层需要保存特征图
        这里 x % i 与 x 等价例如在最后一层: 
        f = [17,20,23] , i = 24 
        y = [x % i for x in ([f] if isinstance(f, int) else f) if x != -1]
        print(y) # [17, 20, 23]
        写成 x % i 可能为了防止 x >= i 的情况，即前面层用到后面层的情况
        """
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist；只存head中的12,16,19,22,24层的输入层的非-1的index[6, 4, 14, 10, 17,20,23]，因为只有这几层的输入层含有非-1的index（注意是存输入层的非-1的index而不是把12,16,19,22,24存起来）
        layers.append(m_)
        if i == 0: # 如果是初次迭代，则新创建一个ch（因为形参ch在创建第一个网络模块时需要用到，所以创建网络模块之后再初始化ch）
            ch = []
        ch.append(c2)
    # 将所有的层封装为nn.Sequential，对保存的特征图排序
    return nn.Sequential(*layers), sorted(save) # [6,4,14,10,17,20,23]->[4,6,10,14,17,20,23]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='yolov5s.yaml', help='model.yaml')
    parser.add_argument('--batch-size', type=int, default=1, help='total batch size for all GPUs')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--profile', action='store_true', help='profile model speed')
    parser.add_argument('--line-profile', action='store_true', help='profile model speed layer by layer')
    parser.add_argument('--test', action='store_true', help='test all yolo*.yaml')
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device) # 生成一个“测试“图片
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / 'models').rglob('yolo*.yaml'):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f'Error in {cfg}: {e}')

    else:  # report fused model summary
        model.fuse()
