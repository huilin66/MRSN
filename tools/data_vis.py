import argparse
import os
# 在最开始设置虚拟显示
os.environ['DISPLAY'] = ':99'
os.system('Xvfb :99 -screen 0 1024x768x24 +extension GLX &')
os.system('sleep 1')  # 等待Xvfb启动
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from spectral import *  # 高光谱处理库
import numpy as np
import rasterio
import scipy.io as sio
import  os
import scipy.io as sio
from spectral import *




def read_downsampled(path, scale):
    """读取并降采样TIFF文件，返回 (bands, height, width)"""
    from skimage import io
    from skimage.transform import rescale
    
    data = io.imread(path)  # (height, width, bands) 或 (height, width)
    
    if data.ndim == 2:
        data = data[np.newaxis, :, :]  # (1, height, width)
    else:
        data = np.transpose(data, (2, 0, 1))  # (bands, height, width)
    
    if scale > 1:
        from scipy.ndimage import zoom
        data = zoom(data, (1, 1/scale, 1/scale), order=1)
    
    return data.astype('float32')

def autoscale(band, pmin=2, pmax=98):
    """自动缩放波段值"""
    if np.all(np.isfinite(band)) and (band.max() == band.min()):
        return band.min(), band.max()
    lo, hi = np.percentile(band[np.isfinite(band)], (pmin, pmax))
    if lo == hi:
        lo, hi = band.min(), band.max()
    return float(lo), float(hi)

def show_tiff_bands(path, cmaps=None, bands=None, downsample=4, pmin=2, pmax=98, 
                    figsize=(15,5), save_path=None):
    """
    显示TIFF波段，可选择不同色带和保存为PNG
    
    Parameters:
    -----------
    path : str
        TIFF文件路径
    cmaps : list
        色带列表
    bands : list
        要显示的波段（1-based）
    downsample : int
        降采样因子
    pmin, pmax : float
        自动缩放的百分位数
    figsize : tuple
        图像大小
    save_path : str
        保存PNG的路径，如果为None则只显示
    """
    data = read_downsampled(path, downsample)
    n_bands = data.shape[0]
    
    if bands is None:
        band_idxs = list(range(1, n_bands+1))
    else:
        # allow 1-based indices in CLI
        band_idxs = [b if b>0 else n_bands+b+1 for b in bands]
    
    cmaps = cmaps or []
    # make sure cmaps list matches number of bands to show (repeat last if needed)
    cmap_list = [cmaps[i] if i < len(cmaps) else 'gray' for i in range(len(band_idxs))]
    
    plt.figure(figsize=figsize)
    n = len(band_idxs)
    
    for i, b in enumerate(band_idxs, start=1):
        arr = data[b-1]
        vmin, vmax = autoscale(arr, pmin=pmin, pmax=pmax)
        ax = plt.subplot(1, n, i)
        im = ax.imshow(arr, cmap=cmap_list[i-1], vmin=vmin, vmax=vmax)
        ax.set_title(f'Band {b} ({cmap_list[i-1]})')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存到: {save_path}")
    else:
        plt.show()
    plt.close()
def extract_rgb(tiff_path, r_band, g_band, b_band, downsample=4, pmin=2, pmax=98, save_path=None):
    """
    从多波段TIFF中提取3个指定波段，合成RGB图像
    
    Parameters:
    -----------
    tiff_path : str
        TIFF文件路径
    r_band : int
        红色通道对应的波段索引（1-based）
    g_band : int
        绿色通道对应的波段索引（1-based）
    b_band : int
        蓝色通道对应的波段索引（1-based）
    downsample : int
        降采样因子（默认4）
    pmin, pmax : float
        自动缩放的百分位数（默认2, 98）
    save_path : str
        保存PNG的路径，如果为None则返回RGB数组
    
    Returns:
    --------
    rgb_image : numpy.ndarray
        RGB图像数组 (H, W, 3)，仅在save_path为None时返回
    """
    # 读取数据
    with rasterio.open(tiff_path) as src:
        if downsample <= 1:
            data = src.read()
        else:
            out_shape = (src.count,
                         int(src.height / downsample),
                         int(src.width / downsample))
            data = src.read(
                out_shape=out_shape,
                resampling=Resampling.average
            )
        data = data.astype('float32')
    
    # 提取RGB波段（转换为0-based索引）
    r_data = data[r_band - 1]
    g_data = data[g_band - 1]
    b_data = data[b_band - 1]
    
    # 对每个波段进行独立拉伸
    def stretch_band(band):
        lo, hi = np.percentile(band[np.isfinite(band)], (pmin, pmax))
        if lo == hi:
            lo, hi = band.min(), band.max()
        return np.clip((band - lo) / (hi - lo), 0, 1)
    
    # 组合RGB并拉伸到[0, 1]
    rgb = np.stack([
        stretch_band(r_data),
        stretch_band(g_data),
        stretch_band(b_data)
    ], axis=-1)
    
    # 保存或返回
    if save_path:
        plt.imsave(save_path, rgb)
        print(f"RGB图像已保存到: {save_path}")
    else:
        return rgb


def tif2mat(tif_path, mat_path, var_name='data'):
    """将TIFF文件转换为MAT文件"""
    with rasterio.open(tif_path) as src:
        data = src.read()  # (bands, height, width)
    
    # 转换为 (height, width, bands)
    data = np.transpose(data, (1, 2, 0)).astype(np.float32)

    
    sio.savemat(mat_path, {var_name: data})
    print(f"已保存: {mat_path}")

def visualize_hyperspectral_cube(path, bands=None, save_path=None):
    """
    使用spectral库绘制高光谱立方体
    
    Parameters:
    -----------
    path : str
        TIFF文件路径
    bands : list
        RGB波段，默认[29, 19, 9]
    save_path : str
        未使用（spectral的view_cube是交互式的）
    """
    if bands is None:
        bands = [29, 19, 9]
    
    # 转换TIFF为MAT
    mat_path = 'temp_cube.mat'
    tif2mat(path, mat_path, var_name='hsi_data')
    
    # 加载MAT数据
    data = sio.loadmat(mat_path)['hsi_data']
    
    # 显示立方体
    spectral.settings.WX_GL_DEPTH_SIZE = 100
    view_cube(data, bands=bands)
    
    # 清理临时文件
    os.remove(mat_path)


def parse_cmap_list(s):
    """解析色带列表"""
    if not s:
        return []
    if isinstance(s, list):
        return s
    return [c.strip() for c in s.split(',') if c.strip()]

def parse_band_list(s):
    """解析波段列表"""
    if not s:
        return None
    if isinstance(s, list):
        return s
    parts = []
    for token in s.split(','):
        token = token.strip()
        if '-' in token:
            a, b = token.split('-', 1)
            parts.extend(list(range(int(a), int(b)+1)))
        else:
            parts.append(int(token))
    return parts

if __name__ == '__main__':
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='可视化TIFF波段，支持不同色带和保存')
    parser.add_argument('tiff', nargs='?', help='多波段TIFF文件路径')
    parser.add_argument('--cmaps', type=str, default='viridis,plasma,gray', 
                       help='色带列表，逗号分隔 (例如: viridis,gray,plasma)')
    parser.add_argument('--bands', type=str, default='1,2,3', 
                       help='要显示的波段 (1-based). 例如: "1,2,3" 或 "1-3"')
    parser.add_argument('--downsample', type=int, default=4, 
                       help='降采样因子 (1=不降采样)')
    parser.add_argument('--pmin', type=float, default=2.0, 
                       help='自动缩放的下百分位数')
    parser.add_argument('--pmax', type=float, default=98.0, 
                       help='自动缩放的上百分位数')
    parser.add_argument('--save', type=str, default=None, 
                       help='保存PNG文件路径')
    parser.add_argument('--cube', action='store_true', 
                       help='启用高光谱立方体可视化')
    
    args = parser.parse_args()
    
    # 如果提供了tiff参数，使用命令行模式
    if args.tiff:
        cmaps = parse_cmap_list(args.cmaps)
        bands = parse_band_list(args.bands)
        
        if args.cube:
            # 高光谱立方体可视化
            visualize_hyperspectral_cube(
                args.tiff, 
                bands=bands, 
                downsample=max(1, args.downsample),
                save_path=args.save
            )
        else:
            # 标准波段可视化
            show_tiff_bands(
                args.tiff, 
                cmaps=cmaps, 
                bands=bands, 
                downsample=max(1, args.downsample),
                pmin=args.pmin, 
                pmax=args.pmax,
                save_path=args.save
            )
    else:
        # 示例：直接处理数据文件夹
        root_dir = r'/scrinvme/huilin/bdd/cp_data/C2Seg/src/C2Seg_BW/train'
        hsi_dir = os.path.join(root_dir, 'hsi')
        msi_dir = os.path.join(root_dir, 'msi')
        sar_dir = os.path.join(root_dir, 'sar')
        label_dir = os.path.join(root_dir, 'label')
        output_dir = 'vis'
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(hsi_dir):
            file_list = os.listdir(hsi_dir)
            if file_list:
                file_name = file_list[0]
                print(f"处理文件: {file_name}")
                
                # 高光谱数据处理
                hsi_path = os.path.join(hsi_dir, file_name)
                save_cube_path = os.path.join(output_dir, f'{file_name}_hsi_cube.png')
                visualize_hyperspectral_cube(
                    hsi_path,
                    save_path=save_cube_path
                )

                # msi数据处理（示例：显示前3个波段）
                msi_path = os.path.join(msi_dir, file_name)
                save_bands_path = os.path.join(output_dir, f'{file_name}_msi_rgb.png')
                extract_rgb(
                    msi_path, 3,2,1,
                    save_path=save_bands_path
                )
                save_bands_path = os.path.join(output_dir, f'{file_name}_msi_nirgb.png')
                extract_rgb(
                    msi_path, 4,2,1,
                    save_path=save_bands_path
                )

                # sar数据处理（示例：显示单波段）
                sar_path = os.path.join(sar_dir, file_name)
                save_sar_path = os.path.join(output_dir, f'{file_name}_sar1.png')
                show_tiff_bands(
                    sar_path,
                    cmaps=['gray'],
                    bands=[1],
                    save_path=save_sar_path
                )
                save_sar_path = os.path.join(output_dir, f'{file_name}_sar2.png')
                show_tiff_bands(
                    sar_path,
                    cmaps=['gray'],
                    bands=[2],
                    save_path=save_sar_path
                )
                
                print("可视化完成！")
            else:
                print("HSI目录为空")
        else:
            print(f"目录不存在: {hsi_dir}")