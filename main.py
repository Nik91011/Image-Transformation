###############
##Design three functions   
# (1) "rot_counter90" to return: image rotated counter clockwise 90 degree, 3x3 rotation matrix to tranbsform uv, roi bounds (x0, y0, x1, y1) in rotated image space
# (2) "crop_roi" to return: image cropped with roi_bounds, 3x3 crop matrix to transform uv to the cropped image space
# (3) "resize" to return: image resized to target_hw, 3x3 resize matrix to transform uv to the resized image space 
# NO change outside these three functions.
# Gradings will be based on the saved images produced by "vis_pts_on_img".
# Do NOT import any library or APIs besides what has been listed. You are only allowed to use the library and libaray function already imported to you.  
###############
import numpy as np
import cv2

#Defination of the four points: eye1, eye2, nose, mouse
org_pts_uv=  np.array([[370, 150], [370, 270], [310, 210], [270, 210]]) #4 points [u v] in 't3face.jpeg'
pts_color = [(255, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255)]
org_roi_bounds = np.array([60, 50, 580, 370]) #[x0, y0, x1, y1] in 't3face.jpeg'
resize_img_hw = (512, 256)


def vis_pts_on_img(image, savefile, transform_mat_list = [], roi_bounds = None):
    image_overlay = np.copy(image)
    image_overlay = np.ascontiguousarray(image_overlay, dtype=np.uint8)
    imgh, imgw = image.shape[:2]
    trans_mat = np.identity(3)
    for mat in transform_mat_list:
       trans_mat = mat @ trans_mat
    org_pts_uv_homo = np.concatenate([org_pts_uv, np.ones_like(org_pts_uv[:,[0]])], axis=1)
    pts_uv_homo =  trans_mat @ org_pts_uv_homo.T
    pts_uv = pts_uv_homo[:2].astype('int').T
    for uv, color in zip(pts_uv.tolist(), pts_color):
        u, v = uv[0], uv[1]
        if max(min(u, imgw), 0) == u and max(min(v, imgh), 0) == v: 
            cv2.circle(image_overlay, (u, v), 1, color, 8, cv2.LINE_AA)
    if roi_bounds is not None:
        x0, y0, x1, y1 = roi_bounds
        if max(min(x0, imgw), 0) == u and max(min(y0, imgh), 0) == v:
            if  max(min(x1, imgw), 0) == u and max(min(y1, imgh), 0) == v:
                cv2.line(image_overlay, (x0, y0), (x0, y1), (0,0,255), 4, cv2.LINE_AA)
                cv2.line(image_overlay, (x0, y1), (x1, y1), (0,0,255), 4, cv2.LINE_AA)
                cv2.line(image_overlay, (x1, y1), (x1, y0), (0,0,255), 4, cv2.LINE_AA)
                cv2.line(image_overlay, (x0, y0), (x1, y0), (0,0,255), 4, cv2.LINE_AA)
    cv2.imwrite(savefile, image_overlay)

def rot_counter90(image, roi_bounds):
    """
    Args: 
    image | np.array() [imgh, imgw, 3]
    roi_bounds(x0, y0, x1, y1)| np.array() [4,]
    Return:
    image rotated counter clockwise 90 degree | np.array() [imgh, imgw, 3]
    rotation  matrix to tranbsform uv | np.array() [3, 3] 
    roi bounds (x0, y0, x1, y1) on image rotated counter clockwise 90 degree | np.array() [4]
    """
    height , width = image.shape[:2]
    center = (width/2, height/2)
    M = cv2.getRotationMatrix2D(center , 90 ,1)
    new_width = int(height * abs(M[0,1]) + width * abs(M[0,0]))
    new_height = int(height * abs(M[0,0]) + width * abs(M[0,1]))
    M[0,2] += (new_width/2) - center[0]
    M[1,2] += (new_height/2) - center[1]
    rotated_image = cv2.warpAffine(image , M , (new_width, new_height))
    x0,y0,x1,y1 = roi_bounds
    roi_bounds_r = np.array([y0, new_width - x1,y1,new_width-x0])
    roi_bounds_r[roi_bounds_r < 0] = 0
    roi_bounds_r[2] = min(roi_bounds_r[2], rotated_image.shape[1])
    roi_bounds_r[3] = min(roi_bounds_r[3], rotated_image.shape[0])
    T_M = np.vstack([M,[0,0,1]])
    print(T_M)
    print(roi_bounds_r)
    return rotated_image, T_M , roi_bounds_r

def crop_roi(image, roi_bounds):
    """
    Args: 
    image | np.array() [imgh, imgw, 3]
    roi_bounds (x0, y0, x1, y1) | np.array() [4,]
    Return:
    image cropped with roi_bounds | np.array() [imgh, imgw, 3]
    crop matrix to transform uv to the cropped image space | np.array() [3, 3] 
    """
    c_y , c_x = np.array(image.shape[:2])//2 
    h_w = min(c_x,c_y)
    n = 120
    m = 80
    roi_bounds = np.array([c_x - h_w + m, c_y - h_w, c_x + h_w - m, c_y + h_w - n])
    x0,y0,x1,y1 = roi_bounds
    c_i = image[y0:y1 , x0:x1]
    T_M_1 = np.array([[1,0,-x0],[0,1,-y0],[0,0,1]])
    print(T_M_1)
    return c_i, T_M_1


def resize(image, target_hw):
    """
    Args: 
    image | np.array() [imgh, imgw, 3]
    target image height and width | tupple (int, int)
    Return:
    image resized to target_hw | np.array() [imgh, imgw, 3]
    resize matrix to transform uv to the resized image space | np.array() [3, 3] 
    """
    t_h , t_w = target_hw
    s = max(t_w/image.shape[1] , t_h/image.shape[0])
    new_w_1 = int(image.shape[1] * s)
    new_h_1 = int(image.shape[0] * s)
    r_i = cv2.resize(image , (new_w_1,new_h_1))
    T_M_2 = np.array([[s,0,0],[0,s,0],[0,0,1]])
    print(T_M_2)
    return r_i, T_M_2

if __name__ == "__main__":
    org_img = cv2.imread('t3face.jpeg')
    vis_pts_on_img(org_img, 't3org.jpeg')
    #
    rot_img, rot_mat, rot_roi_bounds = rot_counter90(org_img, org_roi_bounds)
    vis_pts_on_img(rot_img, 't3rot.jpeg', transform_mat_list=[rot_mat])
    crop_img, crop_mat = crop_roi(rot_img, rot_roi_bounds)
    vis_pts_on_img(crop_img, 't3crop.jpeg', transform_mat_list=[rot_mat, crop_mat])
    resize_img, resize_mat = resize(crop_img, resize_img_hw)
    vis_pts_on_img(resize_img, 't3resize.jpeg', transform_mat_list=[rot_mat, crop_mat, resize_mat])