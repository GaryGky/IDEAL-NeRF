import cv2

if __name__ == '__main__':
    for i in range(6100):
        img_path = f'dataset/May/ori_imgs/{i}.jpg'
        img = cv2.imread(img_path)
        x0, y0 = 00, 650
        img = img[x0:x0 + 650, y0:y0 + 650]
        img = cv2.resize(img, (450, 450), interpolation=cv2.INTER_NEAREST)
        print(img.shape)

        cv2.imwrite(img_path, img)
