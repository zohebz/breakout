#! /usr/bin/python
import gym
import numpy as np
from PIL import Image
import random
import collections
import cv2

def compair_lists(a,b):
	compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
	return compare(a, b)

def compair_unordered_lists(a,b):
	i=0
	if len(a)==len(b):
		while i<len(a):
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

def unordered_lists_diff(a,b):
	#print(a)
	#print(b)
	max_diff = False
	diff_count = 0
	i=0
	if len(a)==len(b):
		while i<len(a):
			if type(a[i]) == type(b[i]) and a[i]!=None and b[i]!=None:
				if a[i]!=b[i]:
					diff_count = diff_count+1	
			else:
				max_diff = True
				break	
			i=i+1
		if not max_diff:
			return diff_count 
	else:
		max_diff = True
	if max_diff:	
		if len(a)>len(b):
			return len(a)
		else:
			return len(b)

def get_pixel_action(curr_frame,prev_frame,row=None):
	#print('row: '+str(row))
	if row!=None:
		frame_diff = np.subtract(curr_frame[row],prev_frame[row])
		i=0
		while i<len(frame_diff):
			if frame_diff[i][0]!=0 or frame_diff[i][1]!=0 or frame_diff[i][2]!=0:
				if curr_frame[row][i][0]!=0 or curr_frame[row][i][1]!=0 or curr_frame[row][i][2]!=0:
					return 3
				else:
					return 2
			i=i+1
	else:
		return None

def get_updated_pixels_list(curr_frame,prev_frame,row=None):
	pixels_list=[]
	if row!=None:
		frame_diff = np.subtract(curr_frame[row],prev_frame[row])
		i=0
		while i<len(frame_diff):
			if frame_diff[i][0]!=0 or frame_diff[i][1]!=0 or frame_diff[i][2]!=0:
				pixels_list.append([row,i])
			i=i+1
	else:
		frame_diff = np.subtract(curr_frame,prev_frame)
		i=0
		while i<len(frame_diff):
			j=0
			while j<len(frame_diff[i]):
				if frame_diff[i][j][0]!=0 or frame_diff[i][j][1]!=0 or frame_diff[i][j][2]!=0:
					pixels_list.append([i][j])
				j=j+1
			i=i+1
	return pixels_list

def get_updated_pixels_color_list(curr_frame,prev_frame,row=None):
	pixels_color_list=[]
	if row!=None:
		frame_diff = np.subtract(curr_frame[row],prev_frame[row])
		i=0
		while i<len(frame_diff):
			if frame_diff[i][0]!=0 or frame_diff[i][1]!=0 or frame_diff[i][2]!=0:
				if curr_frame[row][i][0]!=0 or curr_frame[row][i][1]!=0 or curr_frame[row][i][2]!=0:
					pixels_color_list.append([curr_frame[row][i][0],curr_frame[row][i][1],curr_frame[row][i][2]])	
			i=i+1
	else:
		frame_diff = np.subtract(curr_frame,prev_frame)
		i=0
		while i<len(frame_diff):
			j=0
			while j<(frame_diff[i]):
				if frame_diff[i][j][0]!=0 or frame_diff[i][j][1]!=0 or frame_diff[i][j][2]!=0:
					if curr_frame[i][j][0]!=0 or curr_frame[i][j][1]!=0 or curr_frame[i][j][2]!=0:
						pixels_color_list.append([curr_frame[i][j][0],curr_frame[i][j][1],curr_frame[i][j][2]])
				j=j+1
			i=i+1
	return pixels_color_list

def list_of_lists_to_set(list_of_lists):
	list_to_set = [tuple(lst) for lst in list_of_lists]
	ret_set = set(list_to_set)
	return ret_set 


def generate_random_actions_list(actions_list,count):
	ret_list = []
	for _ in range(count):
		ret_list.append(random.choice(actions_list))
	return ret_list	

def show_learning_view(pixels_set,background):
	apply_color = [0,100,0]
	for pixels in pixels_set:
		background[pixels[0]][pixels[1]] = apply_color
	cv2.imshow('learning view',background)
	cv2.waitKey(100)
	return background

def show_result_view(img_array,background):
	blur_image = cv2.GaussianBlur(img_array, (3,3), 0)
	frame_diff = np.subtract(img_array,blur_image)
	i=0
	while i<len(background):
		j=0
		while j<len(background[i]):
			if frame_diff[i][j][1]>150:
				background[i][j] = [0, 200, 0]
			j=j+1
		i=i+1
	return background

#### enter learnig loop parameters
check_scan_range=input('apply scan range(y/n): ')
from_scan_range_val=None
to_scan_range_val=None
if check_scan_range=='y':
	from_scan_range_val=input('enter from_scan_range: ')
	to_scan_range_val=input('enter to_scan_range (value should be less than 210): ')
####

env = gym.make('Breakout-v0',render_mode='human')

### constant values
actions_list=[2,3]
repeated_values_count = 15 #how many time the values should reapeate to finalize
quit_check_count = 100  
random_actions_count = 10
frame_height = 210
frame_width = 160
accuracy = 3
learning_view = np.zeros((frame_height, frame_width, 3), dtype = "uint8")
expected_result_set = set()
for i in [189,190,191,192]:
	for j in range(144):
		expected_result_set = expected_result_set.union({(i,j+8)}) 
result_view = env.reset() 
######

if from_scan_range_val!=None and to_scan_range_val!=None:
	from_scan_range=int(from_scan_range_val)
	to_scan_range=int(to_scan_range_val)
else:
	from_scan_range=None
	to_scan_range=None

controlled_pixels_dict = {}

row = frame_height - 1
if from_scan_range!=None and to_scan_range!=None:
	upper_limit = to_scan_range
	lower_limit=from_scan_range-1 
else:
	upper_limit = frame_height-1
	lower_limit = -1

row = upper_limit 
print('frame height: '+str(upper_limit-lower_limit)) 
#row=192 #for test
while row>-1:
	print(row)
	rep_ittr_count=0
	exit_count=0
	prev_updated_pixels_set = set()
	prev_updated_pixels_color_set = set()		
	updated_pixels_set = set()
	updated_pixels_color_set = set()	
	while rep_ittr_count<repeated_values_count and exit_count<quit_check_count:
		im_frame_prev = env.reset()
		print('rep_ittr_count: '+str(rep_ittr_count))
		print('exit_count: '+str(exit_count))
		pixel_actions_list = []
		random_actions_list = generate_random_actions_list(actions_list,random_actions_count)
		#print(random_actions_list)
		action_count=0	
		while action_count<random_actions_count:
			observation, reward, done, info = env.step(random_actions_list[action_count])
			im_frame_current = observation
			pixel_action = get_pixel_action(im_frame_current,im_frame_prev,row)
			#print('pixel_action: '+str(pixel_action))
			updated_pixels = get_updated_pixels_list(im_frame_current,im_frame_prev,row)
			updated_pixels = list_of_lists_to_set(updated_pixels)	
			updated_pixels_color = get_updated_pixels_color_list(im_frame_current,im_frame_prev,row)
			updated_pixels_color = list_of_lists_to_set(updated_pixels_color)	
			pixel_actions_list.append(pixel_action)
			updated_pixels_set = updated_pixels_set.union(updated_pixels)
			updated_pixels_color_set = updated_pixels_color_set.union(updated_pixels_color)	
			im_frame_prev = im_frame_current
			action_count=action_count+1	


		#print(pixel_actions_list)
		if unordered_lists_diff(pixel_actions_list,random_actions_list)<accuracy:
			#print('same')
			#print(updated_pixels_set)
			#print(updated_pixels_color_set)
			if updated_pixels_set==prev_updated_pixels_set and updated_pixels_color_set==prev_updated_pixels_color_set:
				rep_ittr_count=rep_ittr_count+1
			else:
				rep_ittr_count=0
		show_learning_view(updated_pixels_set,learning_view)	
		prev_updated_pixels_set = updated_pixels_set
		prev_updated_pixels_color_set = updated_pixels_color_set
		exit_count=exit_count+1	
	if rep_ittr_count==repeated_values_count:
		controlled_pixels_dict[row]={}	
		controlled_pixels_dict[row]['controll_range'] = updated_pixels_set 
		controlled_pixels_dict[row]['colors'] = updated_pixels_color_set 
	row=row-1

print(controlled_pixels_dict)
result_set = set()
for row in controlled_pixels_dict:
	result_set = result_set.union(controlled_pixels_dict[row]['controll_range'])	
save_img_array = show_learning_view(result_set,learning_view)
save_img_array = show_result_view(save_img_array,result_view)
save_frame = Image.fromarray(save_img_array)
save_img_path=input('enter file name to save result: ')
save_frame.save(save_img_path)
print('image saved as '+str(save_img_path))
result_error = len(expected_result_set - result_set)
print('total error: '+str(result_error))

env.close()


		




