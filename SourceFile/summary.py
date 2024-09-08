# --------------------------------------------#
#   该部分代码用于看网络结构
# --------------------------------------------#
import torch
from thop import clever_format, profile
from nets.yolo import YoloBody


def inference_time(module, device):
	# 计算模型的推理时间
	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
	iterations = 300
	# GPU预热
	for _ in range(50):
		_ = module(torch.randn(1, 3, input_shape[0], input_shape[1]).to(device))
	
	# 测速
	times = torch.zeros(iterations)  # 存储每轮iteration的时间
	with torch.no_grad():
		for i in range(iterations):
			starter.record()
			_ = m(torch.randn(1, 3, input_shape[0], input_shape[1]).to(device))
			ender.record()
			# 同步GPU时间
			torch.cuda.synchronize()
			curr_time = starter.elapsed_time(ender)  # 计算时间
			times[i] = curr_time
	
	mean_time = times.mean().item()
	return "Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time)


if __name__ == "__main__":
	input_shape = [640, 640]
	anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	num_classes = 20
	phi = 'l'
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	m = YoloBody(input_shape, num_classes, phi, False).to(device)
	
	infer_time = inference_time(m, device)
	
	dummy_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(device)
	flops, params = profile(m.to(device), (dummy_input,), verbose=False)
	# --------------------------------------------------------#
	#   flops * 2是因为profile没有将卷积作为两个operations
	#   有些论文将卷积算乘法、加法两个operations。此时乘2
	#   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
	#   本代码选择乘2，参考YOLOX。
	# --------------------------------------------------------#
	flops = flops * 2
	flops, params = clever_format([flops, params], "%.3f")
	print('Total GFLOPS: %s' % (flops))
	print('Total params: %s' % (params))
	print(infer_time)
