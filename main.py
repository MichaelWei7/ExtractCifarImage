'''
将图片提取出来
'''
import os, pickle, torch, torchvision

def main():
    file="test_batch"
    image_size=32
    dic=unpickle(file)

    meta=dic[b"batch_label"]
    labels=dic[b"labels"]
    data=dic[b"data"]
    filenames=dic[b"filenames"]

    for i in range(len(labels)):
        label=labels[i]
        filename=filenames[i]
        print(str(filename,"utf-8"))
        data_=data[i].tolist()
        data_ = [data_[i:i+image_size*image_size] for i in range(0, len(data_), image_size*image_size)]

        data_R=data_[0]
        data_G=data_[1]
        data_B=data_[2]

        pixels_R = [data_R[i:i+image_size] for i in range(0, len(data_R), image_size)]
        pixels_G = [data_G[i:i+image_size] for i in range(0, len(data_G), image_size)]
        pixels_B = [data_B[i:i+image_size] for i in range(0, len(data_B), image_size)]

        pixels_R=torch.Tensor(pixels_R).unsqueeze(dim=0)
        pixels_G=torch.Tensor(pixels_G).unsqueeze(dim=0)
        pixels_B=torch.Tensor(pixels_B).unsqueeze(dim=0)
        pixels=torch.cat([pixels_R,pixels_G,pixels_B], dim=0)
        torchvision.utils.save_image(pixels,str(label)+"/"+str(filename,"utf-8") , normalize=True)

def unpickle(file):
    with open(file,"rb") as fo:
        dic=pickle.load(fo, encoding="bytes")
    return dic

if __name__ == '__main__':
    main()
