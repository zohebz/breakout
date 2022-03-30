import gym
import numpy as np
from PIL import Image
import cv2

def find_moving_pixels(current_frame,prev_frame):
    frame_diff = np.subtract(current_frame,prev_frame)
    out_frame = frame_diff
    obj_pixel_list = []
    obj_properties = {'ball':{'width':2},'slider':{'width':16}}
    obj_pixel_dict = {'ball':[],'slider':[]}
    i=0
    while i<len(frame_diff):
        #print('slice')
        j=0
        while j<len(frame_diff[i]):
            #print('frame_diff:'+str(frame_diff[i][j])+' current_frame:'+str(current_frame[i][j]))
            obj_pixels = []
            if frame_diff[i][j][0]!=0 or frame_diff[i][j][1]!=0 or frame_diff[i][j][2]!=0:
                if current_frame[i][j][0]==frame_diff[i][j][0] and current_frame[i][j][1]==frame_diff[i][j][1] and current_frame[i][j][2]==frame_diff[i][j][2] and [i,j] not in obj_pixel_list:
                    obj_pixel_list.append([i,j])
                    obj_pixels.append([i,j])
                    not_zero = True
                    pixel_count_r=0
                    pixel_count_l=0
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
                    not_zero = True
                    while not_zero:
                        if current_frame[i][k][0]!=0 or current_frame[i][k][1]!=0 or current_frame[i][k][2]!=0:
                            obj_pixel_list.append([i,k])
                            obj_pixels.append([i,k])
                            pixel_count_r=pixel_count_r+1
                        else:
                            not_zero=False
                        k=k-1
                    #print(pixel_count_r+pixel_count_l+1)
                    if pixel_count_r+pixel_count_l+1 == obj_properties['ball']['width']:
                        for pixs in obj_pixels:
                            obj_pixel_dict['ball'].append(pixs)
                    elif pixel_count_r+pixel_count_l+1 == obj_properties['slider']['width']:
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
                out_frame[i][j]=[100,0,0]
            elif [i,j] in obj_pixel_dict['slider']:
                out_frame[i][j]=[0,100,0]
            else:
                out_frame[i][j]=[0,0,0]
            j=j+1
        i=i+1

    return out_frame



env = gym.make('Breakout-v0',render_mode='human')
image_path = 'frames/testrgb.png'
observation=env.reset()
im_frame_prev=None
image_count=0
for i_episode in range(100):
    im_frame_current = env.render(mode='rgb_array')
    try:
        if im_frame_prev.any():
            image_array = find_moving_pixels(im_frame_current,im_frame_prev)
    except Exception as e:
        print(e)
        image_array = im_frame_current
        pass
    frame=Image.fromarray(image_array)
    image_count = image_count+1
    frame.save(image_path.split('.')[0]+str(image_count)+'.'+image_path.split('.')[1])
    action = env.action_space.sample() # take a random action
    observation ,reward, done, info = env.step(action)
    im_frame_prev = im_frame_current
    if done:
        print("Episode finished after {} timesteps".format(i_episode+1))
        break
env.close()