import argparse
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import warnings
import skimage.io

# 忽略警告
warnings.filterwarnings('ignore')

def read_tiff_data(path, downsample=1):
    """
    读取TIFF数据并可选降采样
    
    Parameters:
    -----------
    path : str
        TIFF文件路径
    downsample : int
        降采样因子（1=不降采样）
    
    Returns:
    --------
    data : numpy.ndarray
        形状为 (bands, height, width) 的数组
    """
    with rasterio.open(path) as src:
        if downsample <= 1:
            data = src.read()
        else:
            out_shape = (
                src.count,
                int(src.height / downsample),
                int(src.width / downsample)
            )
            data = src.read(
                out_shape=out_shape,
                resampling=Resampling.average
            )
        return data.astype('float32')


def stretch_band(band, pmin=2, pmax=98):
    """
    对单个波段进行百分比拉伸
    
    Parameters:
    -----------
    band : numpy.ndarray
        输入的波段数据
    pmin, pmax : float
        拉伸的百分位数
    
    Returns:
    --------
    stretched : numpy.ndarray
        拉伸到 [0, 1] 范围的数据
    """
    valid_mask = np.isfinite(band)
    if not np.any(valid_mask):
        return np.zeros_like(band)
    
    valid_data = band[valid_mask]
    lo, hi = np.percentile(valid_data, (pmin, pmax))
    
    if lo == hi:
        lo = valid_data.min()
        hi = valid_data.max()
        if lo == hi:  # 如果还是相等，返回全0
            return np.zeros_like(band)
    
    stretched = np.clip((band - lo) / (hi - lo), 0, 1)
    return stretched


def extract_rgb(tiff_path, r_band, g_band, b_band, downsample=4, 
                pmin=2, pmax=98, save_path=None):
    """
    从多波段TIFF中提取3个指定波段，合成RGB图像
    
    Parameters:
    -----------
    tiff_path : str
        TIFF文件路径
    r_band, g_band, b_band : int
        RGB通道对应的波段索引（1-based）
    downsample : int
        降采样因子（默认4）
    pmin, pmax : float
        自动缩放的百分位数（默认2, 98）
    save_path : str
        保存PNG的路径
    
    Returns:
    --------
    rgb_image : numpy.ndarray
        RGB图像数组 (H, W, 3)
    """
    print(f"读取文件: {tiff_path}")
    
    try:
        # 读取数据
        data = read_tiff_data(tiff_path, downsample)
        print(f"数据形状: {data.shape}, 波段数: {data.shape[0]}")
        
        # 检查波段索引是否有效
        max_band = data.shape[0]
        for band_name, band_idx in [('R', r_band), ('G', g_band), ('B', b_band)]:
            if band_idx > max_band or band_idx < 1:
                raise ValueError(f"{band_name}波段索引 {band_idx} 超出范围 [1, {max_band}]")
        
        # 提取RGB波段（转换为0-based索引）
        r_data = data[r_band - 1]
        g_data = data[g_band - 1]
        b_data = data[b_band - 1]
        
        # 对每个波段进行独立拉伸
        r_stretched = stretch_band(r_data, pmin, pmax)
        g_stretched = stretch_band(g_data, pmin, pmax)
        b_stretched = stretch_band(b_data, pmin, pmax)
        
        # 组合RGB
        rgb = np.stack([r_stretched, g_stretched, b_stretched], axis=-1)
        
        # 保存或返回
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.imsave(save_path, rgb)
            print(f"✅ RGB图像已保存到: {save_path}")
        
        return rgb
    
    except FileNotFoundError:
        print(f"❌ 文件不存在: {tiff_path}")
        return None
    except Exception as e:
        print(f"❌ 处理RGB图像时出错: {e}")
        return None


def visualize_single_band(tiff_path, band_idx, cmap='gray', downsample=4,
                          pmin=2, pmax=98, save_path=None):
    """
    可视化单个波段
    
    Parameters:
    -----------
    tiff_path : str
        TIFF文件路径
    band_idx : int
        波段索引（1-based）
    cmap : str
        matplotlib色带名称
    downsample : int
        降采样因子
    pmin, pmax : float
        拉伸的百分位数
    save_path : str
        保存PNG的路径
    
    Returns:
    --------
    band_normalized : numpy.ndarray
        归一化后的波段数据
    """
    print(f"读取文件: {tiff_path}")
    
    try:
        # 读取数据
        data = read_tiff_data(tiff_path, downsample)
        print(f"数据形状: {data.shape}")
        
        # 提取指定波段
        if band_idx > data.shape[0] or band_idx < 1:
            raise ValueError(f"波段索引 {band_idx} 超出范围 [1, {data.shape[0]}]")
        
        band = data[band_idx - 1]
        
        # 拉伸
        band_normalized = stretch_band(band, pmin, pmax)
        
        # 保存
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.imsave(save_path, band_normalized, cmap=cmap)
            print(f"✅ 单波段图像已保存到: {save_path}")
        
        return band_normalized
    
    except FileNotFoundError:
        print(f"❌ 文件不存在: {tiff_path}")
        return None
    except Exception as e:
        print(f"❌ 处理单波段图像时出错: {e}")
        return None


def visualize_multiple_bands(tiff_path, bands=None, cmaps=None, downsample=4,
                            pmin=2, pmax=98, save_path=None):
    """
    可视化多个波段，横向排列显示
    
    Parameters:
    -----------
    tiff_path : str
        TIFF文件路径
    bands : list
        要显示的波段索引列表（1-based）
    cmaps : list
        色带列表，长度与bands相同
    downsample : int
        降采样因子
    pmin, pmax : float
        拉伸的百分位数
    save_path : str
        保存PNG的路径
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        图像对象
    """
    if bands is None:
        bands = [1, 2, 3]
    
    if cmaps is None:
        cmaps = ['gray'] * len(bands)
    elif len(cmaps) < len(bands):
        cmaps.extend(['gray'] * (len(bands) - len(cmaps)))
    
    print(f"读取文件: {tiff_path}")
    
    try:
        # 读取数据
        data = read_tiff_data(tiff_path, downsample)
        print(f"数据形状: {data.shape}")
        
        # 创建图像
        fig, axes = plt.subplots(1, len(bands), figsize=(5*len(bands), 5))
        if len(bands) == 1:
            axes = [axes]
        
        for ax, band_idx, cmap in zip(axes, bands, cmaps):
            if band_idx > data.shape[0] or band_idx < 1:
                print(f"⚠ 波段索引 {band_idx} 超出范围，跳过")
                ax.axis('off')
                continue
            
            band = data[band_idx - 1]
            band_normalized = stretch_band(band, pmin, pmax)
            
            ax.imshow(band_normalized, cmap=cmap)
            ax.set_title(f'Band {band_idx}')
            ax.axis('off')
        
        plt.tight_layout()
        
        # 保存
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ 多波段图像已保存到: {save_path}")
            plt.close()
        else:
            plt.show()
        
        return fig
    
    except FileNotFoundError:
        print(f"❌ 文件不存在: {tiff_path}")
        return None
    except Exception as e:
        print(f"❌ 处理多波段图像时出错: {e}")
        return None

def tif_label_to_color(tif_path, save_path, colormap=None):
    """
    读取单波段TIFF标签图像，映射为彩色图像并保存
    
    Parameters:
    -----------
    tif_path : str
        标签TIFF文件路径（单波段，每个像素值为类别）
    save_path : str
        输出PNG图像路径
    colormap : dict or None
        颜色映射字典，格式 {label_value: (R, G, B)}
        如果为None，使用默认颜色映射
    
    Returns:
    --------
    colored_label : numpy.ndarray
        彩色标签图像 (H, W, 3)
    """
    
    # 读取TIFF标签
    label = skimage.io.imread(tif_path)
    
    print(f"标签形状: {label.shape}")
    print(f"标签唯一值: {np.unique(label)}")
    
    # 默认颜色映射
    if colormap is None:
        colormap = {
            0: (0, 0, 0),           # 背景 - 黑色
            1: (255, 0, 0),         # 类别1 - 红色
            2: (0, 255, 0),         # 类别2 - 绿色
            3: (0, 0, 255),         # 类别3 - 蓝色
            4: (255, 255, 0),       # 类别4 - 黄色
            5: (255, 0, 255),       # 类别5 - 品红
            6: (0, 255, 255),       # 类别6 - 青色
            7: (128, 0, 0),         # 类别7 - 深红
            8: (0, 128, 0),         # 类别8 - 深绿
            9: (0, 0, 128),         # 类别9 - 深蓝
            10: (128, 128, 0),      # 类别10 - 橄榄
        }
    
    # 创建彩色图像
    h, w = label.shape
    colored_label = np.zeros((h, w, 3), dtype=np.uint8)
    
    # 应用颜色映射
    for label_value, color in colormap.items():
        mask = (label == label_value)
        colored_label[mask] = color
    
    # 保存图像
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    skimage.io.imsave(save_path, colored_label)
    print(f"✅ 彩色标签图像已保存到: {save_path}")
    
    return colored_label

def process_modality_pair(hsi_path, msi_path, sar_path, output_dir, file_name, 
                          downsample=4):
    """
    处理一组HSI/MSI/SAR数据并生成可视化
    
    Parameters:
    -----------
    hsi_path : str
        高光谱图像路径
    msi_path : str
        多光谱图像路径
    sar_path : str
        SAR图像路径
    output_dir : str
        输出目录
    file_name : str
        输出文件名前缀
    downsample : int
        降采样因子
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理HSI数据 - 生成伪彩色图像（使用前3个波段）
    if os.path.exists(hsi_path):
        print(f"\n处理HSI: {os.path.basename(hsi_path)}")
        # 尝试读取并显示前3个波段
        try:
            data = read_tiff_data(hsi_path, downsample)
            n_bands = data.shape[0]
            
            # 如果波段数足够，生成RGB
            if n_bands >= 3:
                extract_rgb(
                    hsi_path, 1, 2, 3,
                    downsample=downsample,
                    save_path=os.path.join(output_dir, f'{file_name}_hsi_rgb.png')
                )
            
            # 如果波段数更多，生成近红外伪彩色
            if n_bands >= 4:
                extract_rgb(
                    hsi_path, 4, 3, 2,
                    downsample=downsample,
                    save_path=os.path.join(output_dir, f'{file_name}_hsi_nirgb.png')
                )
            
            # 显示所有波段的概览（如果波段数不多）
            if n_bands <= 10:
                bands_to_show = list(range(1, n_bands + 1))
                visualize_multiple_bands(
                    hsi_path,
                    bands=bands_to_show,
                    cmaps=['viridis'] * n_bands,
                    downsample=downsample,
                    save_path=os.path.join(output_dir, f'{file_name}_hsi_all_bands.png')
                )
        except Exception as e:
            print(f"❌ HSI处理失败: {e}")
    else:
        print(f"⚠ HSI文件不存在: {hsi_path}")
    
    # 处理MSI数据
    if os.path.exists(msi_path):
        print(f"\n处理MSI: {os.path.basename(msi_path)}")
        try:
            data = read_tiff_data(msi_path, downsample)
            n_bands = data.shape[0]
            
            # 自然彩色（RGB）
            if n_bands >= 3:
                extract_rgb(
                    msi_path, 3, 2, 1,
                    downsample=downsample,
                    save_path=os.path.join(output_dir, f'{file_name}_msi_rgb.png')
                )
            
            # 近红外假彩色
            if n_bands >= 4:
                extract_rgb(
                    msi_path, 4, 2, 1,
                    downsample=downsample,
                    save_path=os.path.join(output_dir, f'{file_name}_msi_nirgb.png')
                )
            
            # 彩色红外
            if n_bands >= 4:
                extract_rgb(
                    msi_path, 4, 3, 2,
                    downsample=downsample,
                    save_path=os.path.join(output_dir, f'{file_name}_msi_cir.png')
                )
                
        except Exception as e:
            print(f"❌ MSI处理失败: {e}")
    else:
        print(f"⚠ MSI文件不存在: {msi_path}")
    
    # 处理SAR数据 - 分别显示两个极化通道
    if os.path.exists(sar_path):
        print(f"\n处理SAR: {os.path.basename(sar_path)}")
        try:
            data = read_tiff_data(sar_path, downsample)
            n_bands = data.shape[0]
            
            # 显示前两个波段
            for i in range(min(2, n_bands)):
                visualize_single_band(
                    sar_path,
                    band_idx=i+1,
                    cmap='gray',
                    downsample=downsample,
                    pmin=2, pmax=98,
                    save_path=os.path.join(output_dir, f'{file_name}_sar_band{i+1}.png')
                )
            
            # 如果正好有2个波段，创建双极化伪彩色合成
            if n_bands == 2:
                extract_rgb(
                    sar_path, 1, 2, 1,
                    downsample=downsample,
                    pmin=2, pmax=98,
                    save_path=os.path.join(output_dir, f'{file_name}_sar_dualpol.png')
                )
                
        except Exception as e:
            print(f"❌ SAR处理失败: {e}")
    else:
        print(f"⚠ SAR文件不存在: {sar_path}")


def parse_band_list(s):
    """
    解析波段列表字符串
    
    Examples:
    ---------
    '1,2,3' -> [1, 2, 3]
    '1-3,5' -> [1, 2, 3, 5]
    """
    if not s:
        return None
    if isinstance(s, list):
        return s
    
    parts = []
    for token in s.split(','):
        token = token.strip()
        if '-' in token:
            try:
                a, b = token.split('-', 1)
                parts.extend(list(range(int(a), int(b) + 1)))
            except:
                print(f"⚠ 无法解析范围: {token}")
        else:
            try:
                parts.append(int(token))
            except:
                print(f"⚠ 无法解析数字: {token}")
    
    return parts if parts else None


def parse_cmap_list(s):
    """解析色带列表字符串"""
    if not s:
        return []
    if isinstance(s, list):
        return s
    return [c.strip() for c in s.split(',') if c.strip()]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='多模态遥感数据可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 提取RGB图像（自然彩色）
  python vis_tool.py image.tif --mode rgb --rgb-bands 3,2,1 --save output.png
  
  # 提取RGB图像（近红外假彩色）
  python vis_tool.py image.tif --mode rgb --rgb-bands 4,3,2 --save nirgb.png
  
  # 显示单个波段
  python vis_tool.py image.tif --mode single --band 1 --cmap gray --save band1.png
  
  # 显示多个波段
  python vis_tool.py image.tif --mode multi --bands 1,2,3 --cmaps gray,viridis,plasma
  
  # 批量处理模式（不提供tiff参数）
  python vis_tool.py
        '''
    )
    
    parser.add_argument('tiff', nargs='?', help='TIFF文件路径')
    
    parser.add_argument('--mode', 
                       choices=['rgb', 'single', 'multi', 'batch'],
                       default='rgb',
                       help='可视化模式 (默认: rgb)')
    
    parser.add_argument('--rgb-bands', type=str, default='1,2,3',
                       help='RGB模式下的波段索引，逗号分隔 (默认: 1,2,3)')
    
    parser.add_argument('--band', type=int, default=1,
                       help='单波段模式下的波段索引 (默认: 1)')
    
    parser.add_argument('--bands', type=str, default=None,
                       help='多波段模式下的波段列表，如: 1,2,3 或 1-5')
    
    parser.add_argument('--cmaps', type=str, default='gray',
                       help='色带列表，逗号分隔 (默认: gray)')
    
    parser.add_argument('--downsample', type=int, default=4,
                       help='降采样因子，1为不降采样 (默认: 4)')
    
    parser.add_argument('--pmin', type=float, default=2.0,
                       help='拉伸的最小百分位数 (默认: 2)')
    
    parser.add_argument('--pmax', type=float, default=98.0,
                       help='拉伸的最大百分位数 (默认: 98)')
    
    parser.add_argument('--save', type=str, default=None,
                       help='保存输出图像的路径')
    
    parser.add_argument('--data-dir', type=str, 
                       default=None,
                       help='批量处理模式的数据根目录')
    
    parser.add_argument('--output-dir', type=str, default='vis',
                       help='输出目录 (默认: vis)')
    
    args = parser.parse_args()
    
    # 单文件处理模式
    if args.tiff:
        if not os.path.exists(args.tiff):
            print(f"❌ 文件不存在: {args.tiff}")
            return
        
        if args.mode == 'rgb':
            bands = parse_band_list(args.rgb_bands)
            if len(bands) != 3:
                print(f"❌ RGB模式需要3个波段，但提供了{len(bands)}个: {bands}")
                return
            extract_rgb(args.tiff, *bands,
                       downsample=max(1, args.downsample),
                       pmin=args.pmin, pmax=args.pmax,
                       save_path=args.save)
        
        elif args.mode == 'single':
            visualize_single_band(args.tiff, args.band,
                                 cmap=args.cmaps.split(',')[0].strip(),
                                 downsample=max(1, args.downsample),
                                 pmin=args.pmin, pmax=args.pmax,
                                 save_path=args.save)
        
        elif args.mode == 'multi':
            bands = parse_band_list(args.bands) if args.bands else [1, 2, 3]
            cmaps = parse_cmap_list(args.cmaps)
            visualize_multiple_bands(args.tiff, bands=bands, cmaps=cmaps,
                                    downsample=max(1, args.downsample),
                                    pmin=args.pmin, pmax=args.pmax,
                                    save_path=args.save)
    
    # 批量处理模式
    else:
        root_dir = args.data_dir
        output_dir = args.output_dir
        
        if not os.path.exists(root_dir):
            print(f"❌ 数据根目录不存在: {root_dir}")
            print("请使用 --data-dir 指定正确的目录，或提供单个TIFF文件路径")
            return
        
        hsi_dir = os.path.join(root_dir, 'hsi')
        msi_dir = os.path.join(root_dir, 'msi')
        sar_dir = os.path.join(root_dir, 'sar')
        label_dir = os.path.join(root_dir, 'label')
        
        # 检查HSI目录
        if not os.path.exists(hsi_dir):
            print(f"❌ HSI目录不存在: {hsi_dir}")
            return
        
        file_list = [f for f in os.listdir(hsi_dir) 
                    if f.endswith(('.tif', '.tiff', '.TIF', '.TIFF'))][:1]
        
        if not file_list:
            print(f"❌ HSI目录中没有TIFF文件: {hsi_dir}")
            return
        
        print(f"找到 {len(file_list)} 个文件")
        print(f"输出目录: {output_dir}")
        
        # 处理每个文件
        for idx, file_name in enumerate(file_list, 1):
            print(f"\n{'='*60}")
            print(f"处理文件 {idx}/{len(file_list)}: {file_name}")
            print(f"{'='*60}")
            
            hsi_path = os.path.join(hsi_dir, file_name)
            msi_path = os.path.join(msi_dir, file_name) if os.path.exists(msi_dir) else None
            sar_path = os.path.join(sar_dir, file_name) if os.path.exists(sar_dir) else None
            label_path = os.path.join(label_dir, file_name) if os.path.exists(label_dir) else None
            
            # 检查MSI和SAR目录中的对应文件
            if msi_path and not os.path.exists(msi_path):
                print(f"⚠ MSI文件不存在: {msi_path}")
                msi_path = None
            
            if sar_path and not os.path.exists(sar_path):
                print(f"⚠ SAR文件不存在: {sar_path}")
                sar_path = None
            
            # 生成输出文件名前缀
            name_prefix = os.path.splitext(file_name)[0]
            
            # 处理
            if msi_path and sar_path:
                process_modality_pair(
                    hsi_path, msi_path, sar_path,
                    output_dir, name_prefix,
                    downsample=max(1, args.downsample)
                )
            elif msi_path:
                process_modality_pair(
                    hsi_path, msi_path, None,
                    output_dir, name_prefix,
                    downsample=max(1, args.downsample)
                )
            elif sar_path:
                process_modality_pair(
                    hsi_path, None, sar_path,
                    output_dir, name_prefix,
                    downsample=max(1, args.downsample)
                )
            else:
                print("⚠ 没有找到MSI或SAR文件，仅处理HSI")
                process_modality_pair(
                    hsi_path, None, None,
                    output_dir, name_prefix,
                    downsample=max(1, args.downsample)
                )
            
            # 处理标签
            if label_path:
                tif_label_to_color(
                    label_path,
                    os.path.join(output_dir, f'{name_prefix}_label.png')
                )
            
        print(f"\n{'='*60}")
        print(f"✅ 批量处理完成！输出保存在: {output_dir}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()