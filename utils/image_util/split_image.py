import imageio

img = imageio.imread('results/cross-language/gt.jpg')
crop = img.shape
print(crop)

size = crop[1] / 12
for i in range(12):
    save_img = img[:, int(i * size):int((i + 1) * size), :]
    print(save_img.shape)
    imageio.imwrite(f'results/cross-language/gt/gt{i}.jpg', save_img)
