from torchvision import io
import torch
png_file = '/opt/tiger/minist/datasets/tianchi/train/mask/8.png'

label = io.read_image(png_file)


thresh = 127

left_idx = label < thresh
right_idx = label >= thresh
label[left_idx] = 0
label[right_idx] = 255



label_int = label.to(int)

print(type(label_int) )
print(label_int.shape)

PALETTE = torch.tensor([
    [0, 0, 0], [255, 255, 255], 
])

b = set()
m = dict()
c, w, h = label_int.shape
for j in range(c):
    for k in range(w):
        for l in range(h):
            lbl = int(label[j][k][l])
            if lbl in m:
                m[lbl]+= 1
            else:
                m[lbl] = 1
print(m)



# labels = PALETTE[label.to(int)].permute(2, 0, 1)
# print(labels.shape)