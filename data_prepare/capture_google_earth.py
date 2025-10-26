import os
import xml.etree.ElementTree as ET
import os
import time
import pyautogui

def capture_kml_screenshots_ge_pro(kml_folder, output_folder):
    """
    使用Google Earth Pro内置截图功能保存KML文件视图
    
    :param kml_folder: 包含KML文件的目录路径
    :param output_folder: 截图保存目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有KML文件
    kml_files = [f for f in os.listdir(kml_folder) if f.lower().endswith('.kml')]
    
    if not kml_files:
        print(f"在目录 {kml_folder} 中没有找到KML文件")
        return
    
    # 启动Google Earth Pro
    # try:
    #     os.startfile(r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe")
    # except Exception as e:
    #     print(f"无法启动Google Earth Pro: {e}")
    #     return
    
    print("等待Google Earth Pro启动...")
    time.sleep(5)  # 等待Google Earth完全加载
    output_path = os.path.join(output_folder, f"0000.png")
    
    for kml_file in kml_files:
        kml_path = os.path.join(kml_folder, kml_file)
        base_name = os.path.splitext(kml_file)[0]
        while not os.path.exists(output_path):
            time.sleep(10)
        output_path = os.path.join(output_folder, f"{base_name}.png")
        if os.path.exists(output_path):
            continue
        
        print(f"处理文件: {kml_file}")
        
        try:
            # 打开KML文件
            pyautogui.click(x=1500, y=800)  # 点击"保存图像"
            time.sleep(2)
            pyautogui.hotkey('ctrl', 'o')  # 打开文件对话框
            time.sleep(2)
            
            # 输入文件路径
            pyautogui.write(kml_path)
            time.sleep(1)
            pyautogui.press('enter')
            pyautogui.press('enter')
            time.sleep(30)  # 等待KML加载，可能需要更长时间取决于文件大小
            # pyautogui.click(x=500, y=1300)  # 点击"保存图像"
            # time.sleep(1)            
            # pyautogui.click(x=2500, y=1300)  # 点击"保存图像"
            # time.sleep(1)            
            # pyautogui.click(x=500, y=200)  # 点击"保存图像"
            # time.sleep(1)
            # pyautogui.click(x=2500, y=200)  # 点击"保存图像"
            # time.sleep(1)
            pyautogui.click(x=1200, y=150)  # 点击"保存图像"
            time.sleep(1)       
            # # 在保存对话框中设置文件名和路径
            pyautogui.write(output_path)
            time.sleep(1)
            pyautogui.press('enter')
            pyautogui.press('enter')
            
        except Exception as e:
            print(f"处理文件 {kml_file} 时出错: {e}")
            # 尝试恢复，关闭可能打开的对话框
            pyautogui.press('esc')
            time.sleep(1)
        # break

def process_kml_file(input_path, output_folder):
    """处理单个KML文件"""
    # 获取文件名（不带扩展名）
    filename = os.path.splitext(os.path.basename(input_path))[0]
    
    try:
        # 解析KML文件
        tree = ET.parse(input_path)
        root = tree.getroot()
        
        # 定义命名空间（不同KML文件可能使用不同的命名空间）
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # 修改name元素为文件名
        name_elem = root.find('.//kml:name', ns)
        if name_elem is not None:
            name_elem.text = filename
        
        # 处理所有Placemark的LookAt元素
        for lookat in root.findall('.//kml:LookAt', ns):
            # 修改tilt为0
            tilt_elem = lookat.find('kml:tilt', ns)
            if tilt_elem is not None:
                tilt_elem.text = '0'
            
            # 修改altitude为原来的10倍
            altitude_elem = lookat.find('kml:altitude', ns)
            if altitude_elem is not None:
                altitude_elem.text = '0'

            range_elem = lookat.find('kml:range', ns)
            if range_elem is not None:
                range_elem.text = '1024'

        # 确保输出文件夹存在
        os.makedirs(output_folder, exist_ok=True)
        
        # 构建输出路径
        output_path = os.path.join(output_folder, os.path.basename(input_path))
        
        # 保存修改后的文件
        tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        print(f"处理成功: {input_path} -> {output_path}")
        return True
    
    except Exception as e:
        print(f"处理文件 {input_path} 时出错: {str(e)}")
        return False

def batch_process_kml_files(input_folder, output_folder):
    """批量处理KML文件"""
    # 获取输入文件夹中所有KML文件
    kml_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.kml')]
    
    if not kml_files:
        print(f"在文件夹 {input_folder} 中没有找到KML文件")
        return
    
    print(f"找到 {len(kml_files)} 个KML文件需要处理")
    
    success_count = 0
    for kml_file in kml_files:
        input_path = os.path.join(input_folder, kml_file)
        if process_kml_file(input_path, output_folder):
            success_count += 1
    
    print(f"处理完成，成功处理 {success_count}/{len(kml_files)} 个文件")

def batch_process():
    input_folder = r'D:\intern\university1652-first-key\first-key'
    output_folder = r'D:\intern\data\kml_1024'
    batch_process_kml_files(input_folder, output_folder)


if __name__=='__main__':
    # batch_process()
    kml_folder = r'D:\intern\data\kml_1024'
    output_folder = r'D:\intern\data\image_1024'
    capture_kml_screenshots_ge_pro(kml_folder, output_folder)
