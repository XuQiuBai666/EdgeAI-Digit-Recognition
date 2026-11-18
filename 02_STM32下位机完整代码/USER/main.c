/**
 ******************************************************************************
 * @file    main.c
 * @author  STM32F407 Edge:AI Project
 * @brief   STM32F407ZGT6 边缘AI手写数字识别主程序
 * @version V1.0
 * @date    2026.11.17
 *
 * @硬件平台: 正点原子探索者 STM32F407ZGT6
 * @显示屏:   3.5寸电阻式TFTLCD(XPT2046触摸芯片)
 * @工具链:   Keil?MDK?ARMAC5 / STM32Cube.AI
 *
 ******************************************************************************
 * @功能说明
 * 1. 触摸屏手写轨迹采集，Bresenham直线补全断笔
 * 2. 手写画布预处理：外接矩形裁剪、最近邻缩放、居中、笔画膨胀
 * 3. 调用X?CUBE?AI INT8量化CNN模型做MNIST手写数字推理
 * 4. 串口输出识别结果+置信度；LCD显示识别数字 / No Digit!空画布提示
 * 5. UI包含Clear清空、Recognize识别虚拟按键，KEY0执行触摸校准
 *
 ******************************************************************************
 * @预处理流水线(MCU端)
 * 触摸屏坐标 → 28×28二值画布 → 笔迹外接矩形裁剪
 * → 等比缩放至20×20 → 画布居中 → 笔画膨胀加粗
 * → uint8(0~255)转int8(-128~127) →送入AI网络推理
 *
 ******************************************************************************
 * @全局关键宏定义
 * EMPTY_CANVAS_RET  0xFF  : 空画布返回标记，区分识别结果数字0
 * TARGET_SIZE       20    : MNIST标准目标数字像素尺寸
 * CANVAS_SIZE       280   : LCD上手写绘图区域像素大小
 *
 ******************************************************************************
 * @串口输出配置
 * 波特率:115200  8N1
 * 输出内容:AI初始化状态、空画布提示、Recognized Digit、Confidence置信度
 *
 ******************************************************************************
 * @注意事项
 * 1. 启动文件配置：Stack_Size=0x800(2KB), Heap_Size=0x4000(16KB)
 * 2. network.h / network_data_params.h 为Cube?AI自动生成模型头文件
 * 3. 电阻屏建议首次下载后按KEY0执行触摸校准，消除坐标偏移
 * 4. 本版本预处理：最近邻缩放+单次膨胀，无高斯平滑，保留手写特征
 *
 ******************************************************************************
 */

#include "sys.h"
#include "delay.h"
#include "usart.h"
#include "led.h"
#include "key.h"
#include "lcd.h"
#include "usmart.h"
#include "touch.h"

/********** AI相关头文件 **********/
#include "network.h"
#include "network_data_params.h"
#include <string.h>

/********** 界面布局定义（320×480竖屏） **********/
#define CANVAS_SIZE     280     // 手写区域边长
#define CANVAS_X        20      // 手写区域左上角X
#define CANVAS_Y        120     // 手写区域左上角Y

#define BTN_WIDTH       130     // 功能按钮宽度
#define BTN_HEIGHT      45      // 功能按钮高度
#define BTN_CLEAR_X     20      // 清空按钮左上角X
#define BTN_DETECT_X    170     // 识别按钮左上角X
#define BTN_Y           75      // 功能按钮Y坐标（下移避免与标题重叠）

#define RESULT_Y        420     // 结果显示Y坐标

/********** 推理状态常量 **********/
#define EMPTY_CANVAS_RET    0xFF    // 空画布返回标记，用于区分数字0
#define TARGET_SIZE         20      // MNIST标准数字缩放尺寸

/********** 全局变量 **********/
// AI网络句柄
ai_handle network = AI_HANDLE_NULL;
// 激活缓冲区（4字节对齐，全局分配不占用栈空间）
AI_ALIGNED(4) uint8_t ai_activations[AI_NETWORK_DATA_ACTIVATIONS_SIZE];
// 输入输出缓冲区指针
ai_buffer* input_buffer = NULL;
ai_buffer* output_buffer = NULL;

// 28x28二值化画布（黑底白字，最终送入模型）
u8 mnist_canvas[28][28] = {0};
u8 result_digit = 0;    // 最终识别结果

////////////////////////////////////////////////////////////////////////////////
// 辅助工具函数
// 计算两个数的绝对值
u16 my_abs(u16 x1, u16 x2)
{			 
	if(x1 > x2) return x1 - x2;
	else return x2 - x1;
}

// 在28x28画布上画直线（Bresenham整数算法，补全采样间隔导致的断笔）
static void canvas_draw_line(u8 x0, u8 y0, u8 x1, u8 y1)
{
    u8 dx = my_abs(x1, x0);
    u8 dy = my_abs(y1, y0);
    int sx = x0 < x1 ? 1 : -1;
    int sy = y0 < y1 ? 1 : -1;
    int err = dx - dy;
    u8 x = x0, y = y0;

    while(1)
    {
        if(x < 28 && y < 28)
        {
            mnist_canvas[y][x] = 255;
            // 同步膨胀1像素，加粗笔迹
            if(x > 0)    mnist_canvas[y][x-1] = 255;
            if(y > 0)    mnist_canvas[y-1][x] = 255;
        }
        if(x == x1 && y == y1) break;
        int e2 = 2 * err;
        if(e2 > -dy) { err -= dy; x += sx; }
        if(e2 <  dx) { err += dx; y += sy; }
    }
}

////////////////////////////////////////////////////////////////////////////////
// 电容屏绘图辅助函数
// 画水平线
void gui_draw_hline(u16 x0, u16 y0, u16 len, u16 color)
{
	if(len == 0) return;
	LCD_Fill(x0, y0, x0 + len - 1, y0, color);	
}

// 画实心圆
void gui_fill_circle(u16 x0, u16 y0, u16 r, u16 color)
{											  
	u32 i;
	u32 imax = ((u32)r * 707) / 1000 + 1;
	u32 sqmax = (u32)r * (u32)r + (u32)r / 2;
	u32 x = r;
	gui_draw_hline(x0 - r, y0, 2 * r, color);
	for(i = 1; i <= imax; i++) 
	{
		if((i * i + x * x) > sqmax)
		{
 			if(x > imax) 
			{
				gui_draw_hline(x0 - i + 1, y0 + x, 2 * (i - 1), color);
				gui_draw_hline(x0 - i + 1, y0 - x, 2 * (i - 1), color);
			}
			x--;
		}
		gui_draw_hline(x0 - x, y0 + i, 2 * x, color);
		gui_draw_hline(x0 - x, y0 - i, 2 * x, color);
	}
}  

// 画粗线
void lcd_draw_bline(u16 x1, u16 y1, u16 x2, u16 y2, u8 size, u16 color)
{
	u16 t; 
	int xerr = 0, yerr = 0, delta_x, delta_y, distance; 
	int incx, incy, uRow, uCol; 
	if(x1 < size || x2 < size || y1 < size || y2 < size) return; 
	delta_x = x2 - x1;
	delta_y = y2 - y1; 
	uRow = x1; 
	uCol = y1; 
	if(delta_x > 0) incx = 1;
	else if(delta_x == 0) incx = 0;
	else { incx = -1; delta_x = -delta_x; } 
	if(delta_y > 0) incy = 1; 
	else if(delta_y == 0) incy = 0;
	else{ incy = -1; delta_y = -delta_y; } 
	if(delta_x > delta_y) distance = delta_x;
	else distance = delta_y; 
	for(t = 0; t <= distance + 1; t++)
	{  
		gui_fill_circle(uRow, uCol, size, color);
		xerr += delta_x ; 
		yerr += delta_y ; 
		if(xerr > distance) 
		{ 
			xerr -= distance; 
			uRow += incx; 
		} 
		if(yerr > distance) 
		{ 
			yerr -= distance; 
			uCol += incy; 
		} 
	}  
}   

// 电容屏5点触控颜色表
const u16 POINT_COLOR_TBL[5] = {RED, GREEN, BLUE, BROWN, GRED};

/********** 手写数字推理执行函数 **********/
u8 mnist_inference_run(void)
{
    u8 i, j;
    u8 max_index = 0;
    int8_t max_val;
    int8_t input_buf[784];

    u8 x_min = 28, x_max = 0;
    u8 y_min = 28, y_max = 0;

    // ========== 1. 空画布检测 + 查找笔迹最小外接矩形 ==========
    for(i = 0; i < 28; i++)
    {
        for(j = 0; j < 28; j++)
        {
            if(mnist_canvas[i][j] > 0)
            {
                if(j < x_min) x_min = j;
                if(j > x_max) x_max = j;
                if(i < y_min) y_min = i;
                if(i > y_max) y_max = i;
            }
        }
    }

    // 无任何有效笔迹，返回空画布标记，不再返回数字0
    if(x_min > x_max || y_min > y_max)
    {
        printf("Empty canvas, skip inference\r\n");
        return EMPTY_CANVAS_RET;
    }

    u8 w = x_max - x_min + 1;
    u8 h = y_max - y_min + 1;

    // ========== 2. 等比缩放至20×20以内，对齐MNIST标准 ==========
    u8 new_w, new_h;
    if(w > h)
    {
        new_w = TARGET_SIZE;
        new_h = (h * TARGET_SIZE) / w;
    }
    else
    {
        new_h = TARGET_SIZE;
        new_w = (w * TARGET_SIZE) / h;
    }

    // 计算居中偏移，让数字位于28x28画布正中央
    u8 offset_x = (28 - new_w) / 2;
    u8 offset_y = (28 - new_h) / 2;

    // ========== 3. 最近邻插值缩放 + 居中放置 ==========
    u8 temp_canvas[28][28] = {0};
    for(i = 0; i < new_h; i++)
    {
        for(j = 0; j < new_w; j++)
        {
            // 最近邻插值，纯整数运算，速度快
            u8 src_y = y_min + (i * h) / new_h;
            u8 src_x = x_min + (j * w) / new_w;
            temp_canvas[offset_y + i][offset_x + j] = mnist_canvas[src_y][src_x];
        }
    }

    // ========== 4. 单次笔画膨胀，统一笔触粗细 ==========
    u8 final_canvas[28][28] = {0};
    for(i = 1; i < 27; i++)
    {
        for(j = 1; j < 27; j++)
        {
            if(temp_canvas[i][j]     > 0
            || temp_canvas[i-1][j]   > 0
            || temp_canvas[i+1][j]   > 0
            || temp_canvas[i][j-1]   > 0
            || temp_canvas[i][j+1]   > 0)
            {
                final_canvas[i][j] = 255;
            }
        }
    }

    // ========== 5. 像素值转换为INT8格式，送入模型 ==========
    for(i = 0; i < 28; i++)
    {
        for(j = 0; j < 28; j++)
        {
            input_buf[i * 28 + j] = (int8_t)final_canvas[i][j] - 128;
        }
    }

    // ========== 执行AI推理 ==========
    memcpy(input_buffer->data, input_buf, AI_NETWORK_IN_1_SIZE_BYTES);
    ai_network_run(network, input_buffer, output_buffer);

    // ========== 解析结果，提取类别与置信度 ==========
    int8_t* output_data = (int8_t*)output_buffer->data;
    max_val = output_data[0];
    max_index = 0;
    for(i = 1; i < AI_NETWORK_OUT_1_SIZE; i++)
    {
        if(output_data[i] > max_val)
        {
            max_val = output_data[i];
            max_index = i;
        }
    }

    // 串口打印识别结果与置信度
    printf("Recognized Digit: %d, Confidence: %d\r\n", max_index, max_val);
    return max_index;
}

/********** 界面初始化：绘制完整UI框架 **********/
void UI_Init(void)
{
    LCD_Clear(WHITE);
    
    // 1. 顶部标题
    POINT_COLOR = RED;
    BACK_COLOR = WHITE;
    LCD_ShowString(20, 15, 320, 24, 24, (u8 *)"EdgeAI:Digit Recognition");

    // 2. 清空画布按钮（绿色）
    LCD_Fill(BTN_CLEAR_X, BTN_Y, BTN_CLEAR_X + BTN_WIDTH, BTN_Y + BTN_HEIGHT, GREEN);
    LCD_DrawRectangle(BTN_CLEAR_X, BTN_Y, BTN_CLEAR_X + BTN_WIDTH, BTN_Y + BTN_HEIGHT);
    POINT_COLOR = WHITE;
    BACK_COLOR = GREEN;       // 文字背景与按钮同色，消除白块
    LCD_ShowString(BTN_CLEAR_X + 38, BTN_Y + 14, 100, 20, 16, (u8 *)"Clear");
    
    // 3. 开始识别按钮（蓝色）
    LCD_Fill(BTN_DETECT_X, BTN_Y, BTN_DETECT_X + BTN_WIDTH, BTN_Y + BTN_HEIGHT, BLUE);
    LCD_DrawRectangle(BTN_DETECT_X, BTN_Y, BTN_DETECT_X + BTN_WIDTH, BTN_Y + BTN_HEIGHT);
    POINT_COLOR = WHITE;
    BACK_COLOR = BLUE;        // 文字背景与按钮同色，消除白块
    LCD_ShowString(BTN_DETECT_X + 25, BTN_Y + 14, 100, 20, 16, (u8 *)"Recognize");
    
    // 4. 手写区域边框
    POINT_COLOR = BLACK;
    BACK_COLOR = WHITE;
    LCD_DrawRectangle(CANVAS_X, CANVAS_Y, CANVAS_X + CANVAS_SIZE, CANVAS_Y + CANVAS_SIZE);
    
    // 5. 底部结果显示
    POINT_COLOR = BLUE;
    BACK_COLOR = WHITE;
    LCD_ShowString(70, RESULT_Y, 120, 30, 24, (u8 *)"Result:");
    POINT_COLOR = RED;
    LCD_ShowNum(180, RESULT_Y, 0, 1, 24);
}

/********** 电阻触摸屏手写识别主函数 **********/
void rtp_test(void)
{
    u8 key;
    u8 i = 0;
    u16 x, y;
    u16 cx, cy;
    u8 btn_pressed = 0;
    // 记录上一个画布坐标，用于补全断笔
    u8 last_cx = 0xFF, last_cy = 0xFF;

    while(1)
    {
        key = KEY_Scan(0);
        tp_dev.scan(0);

        if(tp_dev.sta & TP_PRES_DOWN)  // 手指按下/移动
        {
            if(tp_dev.x[0] < lcddev.width && tp_dev.y[0] < lcddev.height)
            {
                x = tp_dev.x[0];
                y = tp_dev.y[0];

                // ========== 点击清空画布按钮 ==========
                if(x >= BTN_CLEAR_X && x <= BTN_CLEAR_X + BTN_WIDTH
                && y >= BTN_Y && y <= BTN_Y + BTN_HEIGHT)
                {
                    if(btn_pressed == 0)
                    {
                        btn_pressed = 1;
                        memset(mnist_canvas, 0, sizeof(mnist_canvas));
                        result_digit = 0;
                        last_cx = 0xFF;
                        last_cy = 0xFF;
                        UI_Init();  // 重绘完整界面
                    }
                    continue;
                }

                // ========== 点击开始识别按钮 ==========
                if(x >= BTN_DETECT_X && x <= BTN_DETECT_X + BTN_WIDTH
                && y >= BTN_Y && y <= BTN_Y + BTN_HEIGHT)
                {
                    if(btn_pressed == 0)
                    {
                        btn_pressed = 1;
                        result_digit = mnist_inference_run();
						
                        // 更新底部结果区域
                        POINT_COLOR = WHITE;
                        BACK_COLOR = WHITE;
                        LCD_Fill(180, RESULT_Y, 280, RESULT_Y + 24, WHITE);
                        
                        if(result_digit == EMPTY_CANVAS_RET)
                        {
                            // 空画布显示提示
                            POINT_COLOR = RED;
                            LCD_ShowString(180, RESULT_Y + 4, 100, 16, 16, (u8 *)"No Digit!");
                        }
                        else
                        {
                            // 正常显示识别数字
                            POINT_COLOR = RED;
                            LCD_ShowNum(180, RESULT_Y, result_digit, 1, 24);
                        }
                    }
                    continue;
                }

                // ========== 在手写区域内绘画 ==========
                if(x >= CANVAS_X && x < CANVAS_X + CANVAS_SIZE
                && y >= CANVAS_Y && y < CANVAS_Y + CANVAS_SIZE)
                {
                    TP_Draw_Big_Point(x, y, RED);

                    // 坐标映射：280px → 28px，10:1等比缩小
                    cx = (x - CANVAS_X) / 10;
                    cy = (y - CANVAS_Y) / 10;

                    if(cx < 28 && cy < 28)
                    {
                        if(last_cx != 0xFF && last_cy != 0xFF)
                        {
                            // 两点之间连线，补全采样间隔导致的断笔
                            canvas_draw_line(last_cx, last_cy, cx, cy);
                        }
                        else
                        {
                            // 按下的第一个点
                            mnist_canvas[cy][cx] = 255;
                            if(cx > 0)    mnist_canvas[cy][cx-1] = 255;
                            if(cy > 0)    mnist_canvas[cy-1][cx] = 255;
                        }
                        last_cx = cx;
                        last_cy = cy;
                    }
                }
            }
        }
        else  // 手指抬起
        {
            btn_pressed = 0;
            // 重置上一点坐标，下次按下重新开始
            last_cx = 0xFF;
            last_cy = 0xFF;
            delay_ms(10);
        }

        // KEY0按下 → 执行触摸校准
        if(key == KEY0_PRES)
        {
            LCD_Clear(WHITE);
            TP_Adjust();
            TP_Save_Adjdata();
            UI_Init();
        }

        i++;
        if(i % 20 == 0) LED0 = !LED0;
    }
}

/********** 电容触摸屏测试函数 **********/
void ctp_test(void)
{
	u8 t = 0;
	u8 i = 0;	  	    
 	u16 lastpos[5][2];
	while(1)
	{
		tp_dev.scan(0);
		for(t = 0; t < 5; t++)
		{
			if((tp_dev.sta) & (1 << t))
			{
				if(tp_dev.x[t] < lcddev.width && tp_dev.y[t] < lcddev.height)
				{
					if(lastpos[t][0] == 0XFFFF)
					{
						lastpos[t][0] = tp_dev.x[t];
						lastpos[t][1] = tp_dev.y[t];
					}
                    
					lcd_draw_bline(lastpos[t][0], lastpos[t][1], tp_dev.x[t], tp_dev.y[t], 2, POINT_COLOR_TBL[t]);
					lastpos[t][0] = tp_dev.x[t];
					lastpos[t][1] = tp_dev.y[t];
				}
			}
            else
            {
                lastpos[t][0] = 0XFFFF;
            }
		}
		
		delay_ms(5);
        i++;
		if(i % 20 == 0) LED0 = !LED0;
	}	
}

int main(void)
{ 
    HAL_Init();                    // 初始化HAL库
    Stm32_Clock_Init(336, 8, 2, 7); // 设置系统时钟168MHz
    delay_init(168);                // 初始化延时函数
    uart_init(115200);              // 初始化调试串口
    usmart_dev.init(84);            // 初始化USMART
    LED_Init();                     // 初始化LED
    KEY_Init();                     // 初始化按键
    LCD_Init();                     // 初始化LCD屏幕
    tp_dev.init();                  // 初始化触摸屏

    /********** AI 模型初始化 **********/
    const ai_handle act_addr[] = { ai_activations };
    ai_error err = ai_network_create_and_init(&network, act_addr, NULL);

    if(err.type != AI_ERROR_NONE)
    {
        printf("AI Init Failed! type:%d code:%d\r\n", err.type, err.code);
        while(1);
    }
    printf("AI Init Success!\r\n");

    input_buffer = ai_network_inputs_get(network, NULL);
    output_buffer = ai_network_outputs_get(network, NULL);
    printf("=== EdgeAI Handwriting Test Start ===\r\n");

    // 初始化完整UI界面
    UI_Init();

    // 根据屏幕类型进入对应测试循环
    if(tp_dev.touchtype & 0X80)
        ctp_test();  // 电容屏
    else
        rtp_test();  // 电阻屏
}
