import pyautogui
import time
import keyboard

print("坐标获取工具已启动！")
print("将鼠标移动到需要获取坐标的位置，然后按下空格键")
print("按ESC键退出")

while True:
    if keyboard.is_pressed('space'):
        x, y = pyautogui.position()
        print(f"当前位置: x={x}, y={y}")
        time.sleep(0.5)  # 防止连续读取
    if keyboard.is_pressed('esc'):
        break
    time.sleep(0.1)

print("坐标获取工具已退出")