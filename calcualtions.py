#! /usr/bin/python
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import gym
import time
from statistics import mean
import random

def find_moving_pixels(current_frame,prev_frame,save_file_path):
	frame_diff = np.subtract(current_frame,prev_frame)
	out_frame = frame_diff
	i=0
	obj_pixel_list = []
	obj_properties = {'ball':{'width':2},'slider':{'width':16}}
	obj_pixel_dict = {'ball':[],'slider':[]}
	while i<len(frame_diff):
		#print('slice')
		j=0
		while j<len(frame_diff[i]):
			#print('frame_diff:'+str(frame_diff[i][j])+' current:'+str(current_frame[i][j]))
			obj_pixels = []	
			if frame_diff[i][j][0]!=0 or frame_diff[i][j][1]!=0 or frame_diff[i][j][2]!=0:
				if current_frame[i][j][0]==frame_diff[i][j][0] and current_frame[i][j][1]==frame_diff[i][j][1] and current_frame[i][j][1]==frame_diff[i][j][1] and [i,j] not in obj_pixel_list: 
					obj_pixel_list.append([i,j])
					obj_pixels.append([i,j])	
					not_zero=True
					pixel_count_r = 0
					pixel_count_l = 0	
					k=j+1
					while not_zero:
						if current_frame[i][k][0]!=0 or current_frame[i][k][1]!=0 or current_frame[i][k][2]!=0:
							obj_pixel_list.append([i,k])
							obj_pixels.append([i,k])
							pixel_count_r=pixel_count_r+1
						else:
							not_zero=False	
						k=k+1	
					k=j-1
					not_zero=True
					while not_zero:
						if current_frame[i][k][0]!=0 or current_frame[i][k][1]!=0 or current_frame[i][k][2]!=0:
							obj_pixel_list.append([i,k])
							obj_pixels.append([i,k])
							pixel_count_l=pixel_count_l+1
						else:
							not_zero=False	
						k=k-1
					#print(pixel_count_l+pixel_count_r+1)
					if pixel_count_l+pixel_count_r+1 == obj_properties['ball']['width']:
						for pixs in obj_pixels: 
							obj_pixel_dict['ball'].append(pixs)	
					elif pixel_count_l+pixel_count_r+1 == obj_properties['slider']['width']:
						for pixs in obj_pixels: 
							obj_pixel_dict['slider'].append(pixs)	
			j=j+1
		i=i+1
	i=0
	print(obj_pixel_dict)
	while i<len(out_frame):
		j=0	
		while j<len(out_frame[i]):
			if [i,j] in obj_pixel_dict['ball']:
				print("ball")
				out_frame[i][j]=[100,0,0]	
			elif [i,j] in obj_pixel_dict['slider']:
				print("slider")
				out_frame[i][j]=[0,100,0]	
			else:
				out_frame[i][j]=[0,0,0]	
			j=j+1
		i=i+1
	im_out_frame = Image.fromarray(out_frame)
	im_out_frame.save(save_file_path)

def get_pixel_action(curr_frame,prev_frame,row=None):
	#print('row: '+str(row))
	if row:
		frame_diff = np.subtract(curr_frame[row],prev_frame[row])
		i=0
		while i<len(frame_diff):
			if frame_diff[i][0]!=0 or frame_diff[i][1]!=0 or frame_diff[i][2]!=0:
				if curr_frame[row][i][0]!=0 or curr_frame[row][i][1]!=0 or curr_frame[row][i][2]!=0:
					return 4
				else:
					return 3
			i=i+1
	else:
		return None

def compair_unordered_lists(a,b):
	i=0
	if len(a)==len(b):
		while i<len(a):
			print(a[i])
			print(b[i])
			if type(a[i]) == type(b[i]): 
				if a[i]!=b[i]:
					return False
			else:
				return False	
			i=i+1
		if i==len(a):
			return True
		else:
			return False	
	else:
		return False

def update_loading_bar_print(value,total_length,content=None,description=None):
	if description == None:
		description=''
	if content==None:
		content=''
	else:
		content='['+content+']'
	bar_print = u"\u2588"*(value+1)
	empty_print = ' '*(total_length-(value+1))
	print_str=description+' |'+bar_print+empty_print+'| '+str(value+1)+'/'+str(total_length)+' '+content
	if value+1==total_length:
		print(print_str)
	else:
		print(print_str, end="\r")



def get_slider_pixels_pos(im_curr_frame,controlled_pixels_dict):
	slider_pixels = []
	pos_range = []
	for row in controlled_pixels_dict:
		for pixel in controlled_pixels_dict[row]['controll_range']:
			if tuple(im_curr_frame[row][pixel[1]]) in list(controlled_pixels_dict[row]['colours']):
				slider_pixels.append([row,pixel[1]])
				pos_range.append(pixel[1])
	pos_range = list(set(pos_range))
	try:
		pos = mean(pos_range)
	except:
		pos = None
	return slider_pixels,pos

def get_to_position(prev_frame,curr_frame ,ignore_pixels):
	pos_range = []
	i=0
	while i<len(prev_frame):
		j=0
		while j<len(prev_frame[i]):
			if prev_frame[i][j][0]!=0 or prev_frame[i][j][1]!=0 or prev_frame[i][j][2]!=0:
				if [i,j] not in ignore_pixels: 
					prev_frame[i][j] = [1,1,1]
					curr_frame[i][j] = [1,1,1]	
			j=j+1
		i=i+1	
	frame_diff = np.subtract(curr_frame,prev_frame)
	i=0
	while i<len(frame_diff):
		j=0
		while j<len(frame_diff[i]):
			if frame_diff[i][j][0]<0 or frame_diff[i][j][1]<0 or frame_diff[i][j][2]<0:
					frame_diff[i][j] = [0,0,0]	
			j=j+1
		i=i+1		
	i=0
	while i<len(frame_diff):
		j=0
		while j<len(frame_diff[i]):
			if frame_diff[i][j][0]!=0 or frame_diff[i][j][1]!=0 or frame_diff[i][j][2]!=0:
				if [i,j] not in ignore_pixels: 
					pos_range.append(j)	
			j=j+1
		i=i+1
	pos_range = list(set(pos_range))
	try:
		pos = mean(pos_range)
	except:
		pos=None
	return pos

def list_of_lists_to_set(list_of_lists):
	list_to_set = [tuple(lst) for lst in list_of_lists]
	ret_set = set(list_to_set)
	return ret_set 

def show_learning_view(pixels_set,background):
	apply_color = [0,255,0]
	black_bg = np.zeros((210, 160, 3), dtype = "uint8")
	for pixels in pixels_set:
		black_bg[pixels[0]][pixels[1]] = apply_color
	gray = cv2.cvtColor(black_bg, cv2.COLOR_BGR2GRAY)
	ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_OTSU)
	contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	result_frame = cv2.drawContours(background, contours, -1,(255,0,255),1)
	cv2.imshow('learning view',result_frame)
	cv2.waitKey(100)	
	return result_frame

def predict_action(prev_frame,curr_frame,controlled_pixels_dict,reach):
	height = 210
	width = 160
	frame_diff = np.zeros((height, width, 3), dtype = "uint8")
	slider_pixels=set()
	moving_pixels = set()
	prev_slider_pixels=set()
	slider_curr_frame = np.copy(curr_frame)
	slider_prev_frame = np.copy(prev_frame)	
	i=0
	#print(controlled_pixels_dict['colors'])
	while i<height:
		j=0
		#print('row '+str(i))
		while j<width:
			if (i,j) in controlled_pixels_dict['pixels']:
				#print('curr_frame '+str(tuple(slider_curr_frame[i][j]))+',prev_frame '+str(tuple(slider_prev_frame[i][j])))
				if tuple(slider_curr_frame[i][j]) in controlled_pixels_dict['colors']:
					slider_pixels = slider_pixels.union({(i,j)}) 
				if tuple(slider_prev_frame[i][j]) in controlled_pixels_dict['colors']:
					prev_slider_pixels = prev_slider_pixels.union({(i,j)}) 
			if prev_frame[i][j][0]!=0 or prev_frame[i][j][1]!=0 or prev_frame[i][j][2]!=0:
				prev_frame[i][j] = [1,1,1]
			if curr_frame[i][j][0]!=0 or curr_frame[i][j][1]!=0 or curr_frame[i][j][2]!=0:	
				curr_frame[i][j] = [1,1,1]
			frame_diff[i][j] = curr_frame[i][j]-prev_frame[i][j]
			if frame_diff[i][j][0]<0 or frame_diff[i][j][1]<0 or frame_diff[i][j][2]<0:
				frame_diff[i][j] = [0,0,0]
			if frame_diff[i][j][0]!=0 or frame_diff[i][j][1]!=0 or frame_diff[i][j][2]!=0:
				moving_pixels = moving_pixels.union({(i,j)})	
			j=j+1
		i=i+1
	print('moving_pixels '+str(moving_pixels)+',prev_slider_pixels '+str(prev_slider_pixels)+',slider_pixels '+str(slider_pixels))		
	moving_pixels = moving_pixels - (prev_slider_pixels.union(slider_pixels))
	slider_pos_range = []
	ball_pos_range = []
	for pixel in slider_pixels:
		slider_pos_range.append(pixel[1])
	for  pixel in moving_pixels:
		ball_pos_range.append(pixel[1])
	print('slider_pos_range '+str(list(set(slider_pos_range)))+',ball_pos_range '+str(list(set(ball_pos_range))))	
	try:
		x_slider_pos = mean(list(set(slider_pos_range)))
		x_ball_pos = mean(list(set(ball_pos_range)))	
	except Exception as e:
		print(e)
		x_slider_pos = None	
		x_ball_pos = None
	print('x_slider_pos '+str(x_slider_pos)+',x_ball_pos '+str(x_ball_pos))
	if x_slider_pos!=None and x_ball_pos!=None:
		if x_slider_pos<x_ball_pos and x_ball_pos-x_slider_pos>reach: 
			return 3
		elif x_slider_pos>x_ball_pos and x_slider_pos-x_ball_pos>reach: 
			return 4
		else:
			return 1	
	else:
		return 1

def get_updated_pixels(curr_frame,prev_frame,row):
	pixels_color_list=[]
	pixels_list = []
	action = None
	if row!=None:
		frame_diff = np.subtract(curr_frame[row],prev_frame[row])
		i=0
		while i<len(frame_diff):
			if frame_diff[i][0]!=0 or frame_diff[i][1]!=0 or frame_diff[i][2]!=0:
				pixels_list.append([row,i])
				if curr_frame[row][i][0]!=0 or curr_frame[row][i][1]!=0 or curr_frame[row][i][2]!=0:
					pixels_color_list.append([curr_frame[row][i][0],curr_frame[row][i][1],curr_frame[row][i][2]])
					action = 4
				else:
					action = 3	
			i=i+1
	else:
		frame_diff = np.subtract(curr_frame,prev_frame)
		i=0
		while i<len(frame_diff):
			j=0
			while j<(frame_diff[i]):
				if frame_diff[i][j][0]!=0 or frame_diff[i][j][1]!=0 or frame_diff[i][j][2]!=0:
					pixels_list.append([row,i])	
					if curr_frame[i][j][0]!=0 or curr_frame[i][j][1]!=0 or curr_frame[i][j][2]!=0:
						pixels_color_list.append([curr_frame[i][j][0],curr_frame[i][j][1],curr_frame[i][j][2]])
				j=j+1
			i=i+1
	return pixels_list,pixels_color_list,action
		

if __name__ == "__main__":

	result_set = set()
	for i in [189,190,191,192]:
		result_set = set()
		colors_set = set()
		for j in range(144):
			result_set = result_set.union({(i,j+8)})
		for colour in [{(200, 72, 72)}]:
			colors_set = colors_set.union(colour)	


	controlled_pixels_dict = {'pixels':result_set,'colors':colors_set}
	print(controlled_pixels_dict)
	env = gym.make('Breakout-v0',render_mode='human')
	
	print('testing....')
	im_prev_frame = env.reset()
	observation, reward, done, info = env.step(1)
	im_curr_frame = observation 
	start_time =time.time()
	reach = 5
	while not done:
		action = predict_action(np.copy(im_prev_frame),np.copy(im_curr_frame),controlled_pixels_dict,reach)
		print(action) 
		observation, reward, done, info = env.step(action)
		im_prev_frame = im_curr_frame 
		im_curr_frame = observation
	env.close()
	print('<-----------------------end------------------------>')
	end_time = time.time()
	total_time = end_time - start_time
	print('total time: '+str(total_time))
