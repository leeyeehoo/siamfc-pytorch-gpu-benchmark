from PIL import Image, ImageStat
import numpy as np

def convert_bbox_format(bbox, to = 'center-based'):
    x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
    if to == 'top-left-based':
        x -= get_center(target_width)
        y -= get_center(target_height)
    elif to == 'center-based':
        y += get_center(target_height)
        x += get_center(target_width)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(to))
    return Rectangle(x*1.0, y*1.0, target_width*1.0, target_height*1.0)
def get_center(x):
    return (x - 1.) / 2.
def get_zbox(bbox, p_rate =0.25):
    x, y, target_width, target_height = bbox.x, bbox.y, bbox.width, bbox.height
    p = 2*p_rate*(target_width+target_height)
    target_sz = np.sqrt(np.prod((target_width+p)*(target_height+p)))
    return Rectangle(x , y, target_sz , target_sz)
def get_xbox(zbox,dx = 0,dy = 0,padding_rate = 1):
    x, y, target_width, target_height = zbox.x+dx*0.5*zbox.width, zbox.y+dy*0.5*zbox.height, zbox.width, zbox.height
    return Rectangle(x , y, target_width*2*padding_rate , target_height*2*padding_rate) 
def gen_xz( img, inbox,to = 'x',pdrt = 1):
    box = Rectangle(inbox.x,inbox.y,inbox.width*pdrt,inbox.height*pdrt)
    x_sz = (255,255)
    z_sz = (127,127)
    bg = Image.new('RGB',(int(box.width),int(box.height)) ,tuple(map(int,ImageStat.Stat(img).mean)))
    bg.paste(img, (-int(box.x -0.5*box.width) ,-int(box.y - 0.5*box.height)))
    if to == 'x':
        temp = bg.resize(x_sz)
    elif to =='z':
        temp = bg.resize(z_sz)
    else:
        raise ValueError("Bbox format: {} was not recognized".format(bbox_type))
    return temp
