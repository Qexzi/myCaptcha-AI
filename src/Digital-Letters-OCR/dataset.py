import os
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms # 目的是将图片转换为张量
from PIL import Image # 用于打开图片
from PIL import ImageFilter

class myDataset(Dataset):
    def __init__(self, root_dir, train=True, file_list=None):
        """
        :param root_dir: 图片目录。传 str=单个目录;传 list/tuple=多个目录(自动合并)
        :param train: True=启用数据增强;False=不增强
        :param file_list: 若提供,直接使用该图片路径列表(跳过目录扫描),用于微调的真实图过采样
        """
        super(myDataset, self).__init__()
        self.train = train # True:训练集,启用数据增强; False:测试集,不增强
        self.transform = ResizeNormalize((64, None), train=train) # 高度64固定
        if file_list is not None:
            all_paths = list(file_list)  # 显式路径列表(微调用)
        else:
            dirs = root_dir if isinstance(root_dir, (list, tuple)) else [root_dir]
            all_paths = []
            for d in dirs:
                all_paths.extend(os.path.join(d, img) for img in os.listdir(d))
        # P0-3: 多来源合并的大数据集难免混入损坏图片,先全量校验剔除,防止训练中途崩溃
        self.images_path = self._filter_valid(all_paths)

    @staticmethod
    def _filter_valid(paths):
        """逐张强制解码校验(open + convert + load),剔除损坏图片,返回可用路径列表。

        注意 Image.open 是惰性读取,只解析头部,像素损坏的文件要等 resize/load
        时才报错,因此这里必须 convert('RGB').load() 强制完整解码,与训练时的
        预处理路径保持一致。
        """
        valid = []
        invalid = []
        total = len(paths)
        bar_w = 30  # 进度条宽度(字符)
        for i, p in enumerate(paths):
            try:
                with Image.open(p) as img:
                    img.convert('RGB').load()  # 强制完整解码像素
                valid.append(p)
            except Exception as e:
                invalid.append((p, e))
            if total and ((i + 1) % 1000 == 0 or i + 1 == total):  # 每 1000 张刷新一次进度条
                pct = (i + 1) / total
                filled = int(bar_w * pct)
                bar = '█' * filled + '─' * (bar_w - filled)
                print(f'\r  校验 [{bar}] {pct*100:5.1f}% {i+1:>8,}/{total:,}', end='', flush=True)
        if total:
            print()  # 进度条结束后换行,避免与后续输出挤在同一行
        if invalid:
            print(f'[警告] 发现 {len(invalid)} 张无法读取的图片,已自动剔除(共 {len(paths)} 张):')
            for p, e in invalid[:10]:
                print(f'    {os.path.basename(p)}: {e}')
            if len(invalid) > 10:
                print(f'    ... 其余 {len(invalid) - 10} 张略')
        return valid

    def __len__(self):
        return len(self.images_path) # 返回数据集的大小

    def __getitem__(self, idx):
        image_path = self.images_path[idx] # 返回第idx个样本的路径
        # print(image_path)
        label = os.path.basename(image_path).split('_')[0]  # 提取图片中的答案 -> 之后要编码才可以被识别(os.path.basename 兼容 Windows 的 '\')
        # print(label)
        with Image.open(image_path) as img: # with 关闭文件句柄,避免 10 万级文件句柄/文件锁堆积
            image = self.transform(img)
        return image_path, label, image
        

class ResizeNormalize(object):
    def __init__(self, size, interpolation=Image.BILINEAR, train=False):
        """
        :param size: 目标尺寸，传 (height, width) 表示固定高宽；传 (height, None) 表示固定高度、宽度按原图比例缩放
        :param interpolation: 插值方式，默认双线性
        :param train: True 时启用数据增强(旋转/缩放抖动/噪声/模糊),False 时只做缩放归一化
        """
        self.size = size
        self.interpolation = interpolation
        self.train = train
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.convert('RGB') # 统一为 3 通道，与模型 nc=3 对齐(若要灰度改 'L',并同步把模型的 nc 改为 1)

        # ---------- 数据增强(仅训练时启用) ----------
        if self.train:
            # 随机水平缩放抖动: 宽度方向乘以 [0.95, 1.05]
            scale_x = random.uniform(0.95, 1.05)
            w, h = img.size
            img = img.resize((int(w * scale_x), h), self.interpolation)

            # 随机旋转 ±5°
            angle = random.uniform(-5, 5)
            img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))

            # 随机高斯模糊(小概率)
            if random.random() < 0.2:
                img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.8)))
        # --------------------------------------------

        if self.size[1] is None: # (height, None) -> 固定高度，宽度按比例
            w, h = img.size
            new_h = self.size[0]
            new_w = int(new_h * w / h) if h > 0 else w
            img = img.resize((new_w, new_h), self.interpolation)
        else: # PIL.resize 期望 (width, height),传入的 size 是 (height, width),需翻转
            img = img.resize((self.size[1], self.size[0]), self.interpolation)
        img = self.toTensor(img) # 将图像转换为张量的过程中自动有[0,255] ->[0,1]
        img.sub_(0.5).div_(0.5)   # in-place 操作，等价于 (img - 0.5) / 0.5 正则化到[-1,1]

        # 随机加噪(训练时,张量层面): 高斯噪声,幅度 ±0.05
        if self.train:
            noise = torch.randn(img.size()) * 0.05
            img = img + noise
            img = img.clamp(-1.0, 1.0)
        return img



if __name__ == '__main__':


    # 获取脚本所在目录，再拼接相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '../../data/data/train/')
    dataset = myDataset(data_dir)
    image_path, label, image = dataset[0]
    print(image, image.shape)

    pass


