import os
import numpy as np
from tqdm import tqdm
from utils import pc_util
from scipy.misc import imsave

def write_result():
    root_path = "/home/lqyu/server/proj49/PointSR_data/test_data/our_collected_data"
    model_names = ['1024_nonormal_generator2_2', '1024_nonormal_generator2_2_uniformloss',
                   '1024_nonormal_generator2_2_recursive']

    index_path = os.path.join("index.html")
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")
    index.write("<th width='5%%'>name</th>")

    index.write("<tr><th></th>")
    for model in model_names:
        index.write("<th>%s</th>" % model)
    index.write("</tr>")

    # get sample list
    items = os.listdir(root_path + "/" + model_names[0])
    items.sort()

    # mkdir model image path
    for model in model_names:
        if not os.path.exists(root_path + "/" + model + "_three_view_img/"):
            os.makedirs(root_path + "/" + model + "_three_view_img/")

    # write img to file
    for item in tqdm(items):
        index.write("<tr>")
        index.write("<td>%s</td>" % item)

        # write prediction
        for model in model_names:
            path = root_path + "/" + model +"/" + item
            if not os.path.exists(path):
                continue
            img_path = root_path + "/" + model + "_three_view_img/" + item
            img_path = img_path.replace("xyz", "png")
            if not os.path.exists(img_path):
                data = np.loadtxt(path)
                data = data[:, 0:3]
                img = pc_util.point_cloud_three_views(data, diameter=8)
                imsave(img_path, img)
            index.write("<td><img width='100%%', src='%s'></td>" % img_path)
        index.write("</tr>")
    index.close()


def write_result2html_benchmark():
    root_path = "/home/lqyu/server/proj49/PointSR_data/test_data/our_collected_data"
    phase = 'surface_benchmark'
    input_path ="../data/"+phase+"/1024_nonuniform"
    gt_path = "../data/"+phase+"/4096"
    model_names = ['1024_nonormal_generator2_2','1024_nonormal_generator2_2_uniformloss','1024_nonormal_generator2_2_recursive']


    index_path = os.path.join(root_path, phase + "_index.html")
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")
    index.write("<th width='5%%'>name</th><th>Input</th>")
    index.write("<th>Refered GT</th></tr>")

    index.write("<tr><th></th>")
    for model in model_names:
        index.write("<th>%s</th>" % model)
    index.write("</tr>")

    # get sample list
    items = os.listdir(root_path + "/" + model_names[0] + "/result/" + phase)
    items.sort()

    # mkdir model image path
    for model in model_names:
        if not os.path.exists(root_path + "/" + model + "/result/" + phase + "_three_view_img/"):
            os.makedirs(root_path + "/" + model + "/result/" + phase + "_three_view_img/")

    # write img to file
    for item in tqdm(items):
        index.write("<tr>")
        index.write("<td>%s</td>" % item)

        # write input image
        object = item.split("_")[0]
        id = item.split(".")[0]
        path = input_path + "/%s.xyz" % (id)
        img_path = input_path + "_three_view_img/%s.png" % (id)
        if not os.path.exists(input_path + "_three_view_img/"):
            os.makedirs(input_path + "_three_view_img/")
        if not os.path.exists(img_path):
            data = np.loadtxt(path)
            data = data[:, 0:3]
            img = pc_util.point_cloud_three_views(data,diameter=8)
            imsave(img_path, img)
        index.write("<td><img width='100%%', src='%s'></td>" % img_path)
        # write gt image
        path = gt_path + "/%s.xyz" % (id)
        img_path = gt_path + "_three_view_img/%s.png" % (id)
        if not os.path.exists(gt_path + "_three_view_img/"):
            os.makedirs(gt_path + "_three_view_img/")
        if not os.path.exists(img_path):
            data = np.loadtxt(path)
            data = data[:, 0:3]
            img = pc_util.point_cloud_three_views(data,diameter=8)
            imsave(img_path, img)
        index.write("<td><img width='100%%', src='%s'></td>" % img_path)
        index.write("</tr>")

        index.write("<tr><th></th>")
        # write prediction
        for model in model_names:
            path = root_path + "/" + model + "/result/" + phase + "/" + item
            if not os.path.exists(path):
                continue
            img_path = root_path + "/" + model + "/result/" + phase + "_three_view_img/" + item
            img_path = img_path.replace("xyz", "png")
            if not os.path.exists(img_path):
                data = np.loadtxt(path)
                data = data[:, 0:3]
                img = pc_util.point_cloud_three_views(data,diameter=8)
                imsave(img_path, img)
            index.write("<td><img width='100%%', src='%s'></td>" % img_path)
        index.write("</tr>")
    index.close()


def write_result2html_ModelNet():
    root_path = "../model"
    gt_path = "../data/ModelNet10_poisson_normal"
    #gt_path = "../data/Patches"
    model_names = ['1024_generator2_2','new_1024_generator2_2','new_1024_generator2_2_fixed_lr']
    phase = 'test'

    index_path = os.path.join(root_path, phase + "_index.html")
    index = open(index_path, "w")
    index.write("<html><body><table><tr>")
    index.write("<th width='5%%'>name</th><th>Input</th>")
    index.write("<th>Refered GT</th></tr>")

    index.write("<tr><th></th>")
    for model in model_names:
        index.write("<th>%s</th>" % model)
    index.write("</tr>")

    # get sample list
    items = os.listdir(root_path + "/" + model_names[0] + "/result/" + phase)
    items.sort()

    # mkdir model image path
    for model in model_names:
        if not os.path.exists(root_path + "/" + model + "/result/" + phase + "_three_view_img/"):
            os.makedirs(root_path + "/" + model + "/result/" + phase + "_three_view_img/")

    # write img to file
    for item in tqdm(items[::25]):
        index.write("<tr>")
        index.write("<td>%s</td>" % item)

        # write input image
        object = item.split("_")[0]
        id = item.split(".")[0]
        fixed = "%s/1024_nonuniform/%s" % (gt_path, 'train')
        path = fixed + "/%s.xyz" % (id)
        img_path = fixed + "_three_view_img/%s.png" % (id)
        if not os.path.exists(fixed + "_three_view_img/"):
            os.makedirs(fixed + "_three_view_img/")
        if not os.path.exists(img_path):
            data = np.loadtxt(path)
            data = data[:, 0:3]
            img = pc_util.point_cloud_three_views(data,diameter=8)
            imsave(img_path, img)
        index.write("<td><img width='100%%', src='%s'></td>" % img_path)
        # write gt image
        fixed = "%s/4096/%s" % (gt_path, 'train')
        path = fixed + "/%s.xyz" % (id)
        img_path = fixed + "_three_view_img/%s.png" % (id)
        if not os.path.exists(fixed + "_three_view_img/"):
            os.makedirs(fixed + "_three_view_img/")
        if not os.path.exists(img_path):
            data = np.loadtxt(path)
            data = data[:, 0:3]
            img = pc_util.point_cloud_three_views(data,diameter=8)
            imsave(img_path, img)
        index.write("<td><img width='100%%', src='%s'></td>" % img_path)
        index.write("</tr>")

        index.write("<tr><th></th>")
        # write prediction
        for model in model_names:
            path = root_path + "/" + model + "/result/" + phase + "/" + item
            if not os.path.exists(path):
                continue
            img_path = root_path + "/" + model + "/result/" + phase + "_three_view_img/" + item
            img_path = img_path.replace("xyz", "png")
            if not os.path.exists(img_path):
                data = np.loadtxt(path)
                data = data[:, 0:3]
                img = pc_util.point_cloud_three_views(data,diameter=8)
                imsave(img_path, img)
            index.write("<td><img width='100%%', src='%s'></td>" % img_path)
        index.write("</tr>")
    index.close()

if __name__ == '__main__':
    write_result2html_ModelNet()
    #write_result2html_benchmark()
    #calculate_emd_error('ModelNet40')
