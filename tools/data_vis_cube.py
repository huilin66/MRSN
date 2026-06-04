"""
高光谱数据立方体可视化并自动截图
支持 .mat 和 .tif/.tiff 格式
"""
import os
import sys
import argparse
import numpy as np
import rasterio
from spectral import *
import skimage

def load_mat(mat_path, key='cube'):
    """加载MAT文件中的高光谱数据"""
    try:
        import scipy.io as sio
        data = sio.loadmat(mat_path)
        if key in data:
            return data[key]
        # 如果没有指定key，尝试找第一个3D数组
        for k, v in data.items():
            if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 3:
                print(f"使用键: {k}")
                return v
        raise KeyError(f"未找到键 '{key}' 或任何3D数组")
    except Exception as e:
        print(f"使用 mat73 加载...")
        try:
            import mat73
            data = mat73.loadmat(mat_path)
            if key in data:
                return data[key]
            for k, v in data.items():
                if not k.startswith('__') and isinstance(v, np.ndarray) and v.ndim == 3:
                    print(f"使用键: {k}")
                    return v
        except:
            raise ValueError(f"无法加载MAT文件: {e}")


def load_tiff(tiff_path, downsample=4):
    """加载TIFF文件并转换为 (H, W, Bands) 格式"""
    with rasterio.open(tiff_path) as src:
        if downsample <= 1:
            data = src.read()
        else:
            out_shape = (
                src.count,
                int(src.height / downsample),
                int(src.width / downsample)
            )
            from rasterio.enums import Resampling
            data = src.read(out_shape=out_shape, resampling=Resampling.average)
    
    # 转换为 (H, W, Bands)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)
    return data


def auto_cube_screenshot(img_path, bands=None, save_path='hsi_cube.png', delay=5000):
    """
    显示高光谱立方体并自动截图
    
    Parameters:
    -----------
    data : numpy.ndarray
        高光谱数据 (H, W, Bands)
    bands : list
        要显示的RGB波段，默认 [29, 19, 9]
    save_path : str
        截图保存路径
    delay : int
        截图延迟（毫秒），默认5000
    """
    import wx
    
    if bands is None:
        bands = [29, 19, 9]
    print(img_path)
    data = skimage.io.imread(img_path)
    # 检查波段有效性
    max_band = data.shape[2]
    valid_bands = [b for b in bands if 1 <= b <= max_band]
    if len(valid_bands) < 3:
        print(f"波段 {bands} 超出范围 [1, {max_band}]")
        valid_bands = list(range(1, min(4, max_band+1)))
        print(f"使用波段: {valid_bands}")
    
    print(f"数据形状: {data.shape}")
    print(f"显示波段: {valid_bands}")
    
    # 设置OpenGL
    settings.WX_GL_DEPTH_SIZE = 16
    
    # 创建应用
    app = wx.App()
    
    # 显示立方体
    window = view_cube(data, bands=valid_bands)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    def save_and_exit():
        """截图并退出"""
        try:
            print(f"截图保存至: {save_path}")
            window.save_screenshot(save_path)
            print(f"✅ 截图成功!")
        except Exception as e:
            print(f"❌ 截图失败: {e}")
        finally:
            window.Close()
            app.Exit()
    
    # 设置定时截图
    print(f"将在 {delay/1000:.1f} 秒后自动截图...")
    wx.CallLater(delay, save_and_exit)
    
    # 运行
    app.MainLoop()
    
    # 验证
    if os.path.exists(save_path):
        print(f"文件已保存: {save_path} ({os.path.getsize(save_path)} bytes)")
    else:
        print("⚠ 文件未生成")



    
def view_cube_only(img_path, bands=None, downsample=4):
    """
    只显示高光谱立方体，不自动截图
    
    Parameters:
    -----------
    img_path : str
        输入文件路径
    bands : list
        要显示的RGB波段，默认 [29, 19, 9]
    downsample : int
        TIFF降采样因子
    """
    import wx
    
    if bands is None:
        bands = [110, 64, 44]
    
    # 加载数据
    data = skimage.io.imread(img_path)
    
    # 确保数据是3D数组 (H, W, Bands)
    if data.ndim == 2:
        data = data[:, :, np.newaxis]
    elif data.ndim == 3 and data.shape[0] < data.shape[2]:
        print(f"检测到数据格式为 (Bands, H, W)，正在转换...")
        data = np.transpose(data, (1, 2, 0))
    
    # 检查波段有效性
    max_band = data.shape[2]
    valid_bands = [b for b in bands if 1 <= b <= max_band]
    if len(valid_bands) < 3:
        print(f"波段 {bands} 超出范围 [1, {max_band}]")
        valid_bands = list(range(1, min(4, max_band+1)))
        print(f"使用波段: {valid_bands}")
    
    print(f"数据形状: {data.shape}")
    print(f"显示波段: {valid_bands}")
    
    # 设置OpenGL和白色背景（必须在view_cube之前设置）
    settings.WX_GL_DEPTH_SIZE = 16
    
    # 创建应用并显示
    app = wx.App()
    view_cube(data, bands=valid_bands, background=(1, 1, 1))  # 直接在view_cube中设置背景色
    app.MainLoop()

if __name__ == '__main__':
   root_dir = None
   data_name = os.listdir(root_dir)[0]
   data_path = os.path.join(root_dir, data_name)
   save_path = os.path.join(root_dir, data_name.replace('.mat', '_cube.png').replace('.tif', '_cube.png'))
#    auto_cube_screenshot(data_path, save_path=save_path, delay=5000)

   view_cube_only(data_path)