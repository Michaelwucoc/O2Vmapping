import torch
import open_clip
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator


# from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class Clipper(object):

    def __init__(self, cfg, args, slam) -> None:

        self.device = cfg['clip']['device']
        self.clip_model_path = cfg['clip']['clip_model_path']
        self.sam_model_path = cfg['clip']['sam_model_path']

        self.clip_model, _, self.preprocess, self.tokenizer = self.clip_init(self.clip_model_path)
        # self.sam_predictor = self.sam_init_predictor(self.sam_model_path, self.device)

        self.mask_generator = self.sam_entire_image_init(self.sam_model_path)

    def clip_init(self, mode_path):

        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=mode_path)
        tokenizer = open_clip.get_tokenizer('ViT-B-16')

        return model, _, preprocess, tokenizer

    def sam_init_predictor(self, sam_model_path, device):

        model_type = 'vit_b'

        sam = sam_model_registry[model_type](checkpoint=sam_model_path)
        sam.to(device)

        predictor = SamPredictor(sam)

        return predictor

    def simple_point_segment(self, input_point, input_label, image):

        '''
            input : point coordinate, label, image
            return : masks(3), scores(3), logits(3)
        '''

        self.sam_predictor.set_image(image)

        masks, scores, logits = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )

        return masks, scores, logits

    def clip_image_feature(self, images):

        """
            images : must be list [image(PIL type)]
            return image feature : (n, 512)
        """

        # change type
        images_PIL = []
        for image in images:
            image = image.astype(np.uint8)
            image = Image.fromarray(image)
            images_PIL.append(image)

        images_torch = [self.preprocess(image) for image in images_PIL]
        images_torch = torch.stack(images_torch)

        # simple image
        if len(images_torch.shape) < 4:
            images_torch = images_torch.unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():

            # not normal
            image_features = self.clip_model.encode_image(images_torch)

        return image_features

    def clip_feature_text(self, features, text):

        """
            input : mask features and query text
            return (features, text_probs) : tensor
        """

        text_token = self.tokenizer(text)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = features
            text_features = self.clip_model.encode_text(text_token)
            image_features /= (image_features.norm(dim=-1, keepdim=True) + 1e-5)
            text_features /= (text_features.norm(dim=-1, keepdim=True))

            text_features = text_features.to(self.device)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            print("Label probs:", text_probs)

        return text_probs

    def sam_entire_image_init(self, sam_model_path):
        sam = sam_model_registry['vit_b'](checkpoint=sam_model_path)
        # sam = sam_model_registry['vit_t'](checkpoint = sam_model_path)
        # sam = sam_model_registry['default'](checkpoint = sam_model_path)
        sam.to(self.device)
        sam.eval()

        mask_generator = SamAutomaticMaskGenerator(sam)
        return mask_generator

    def get_entire_image_mask(self, mask_generator, image_np):

        print('start sam ...')
        masks = mask_generator.generate(image_np)
        print('end sam ! ')
        return masks

    def get_segment_masks(self, masks):

        '''
            Take out the segment_mask from the mask and concatenate all the masks together.
        '''
        masks_np = []
        for mask in masks:
            masks_np.append(mask['segmentation'])

        masks_np = np.stack(masks_np, axis=0)
        return masks_np

    def simple_feature_from_masks(self, masks_np, i, j, orig_shape=None):
        """
        Retrieve the mask values at pixel coordinates (i, j) safely without index errors.

        Args:
            masks_np (np.ndarray): masks array of shape (N_masks, H_mask, W_mask), dtype=bool or uint8.
            i (np.ndarray): y-coordinates (vertical) of pixels, shape (num_points,) or similar.
            j (np.ndarray): x-coordinates (horizontal) of pixels, shape (num_points,) or similar.
            orig_shape (tuple or None): (H_orig, W_orig) original image size, optional.
                                    If provided, will scale i, j from original image coords to mask coords.

        Returns:
            np.ndarray: Boolean or uint8 array of shape (N_masks, num_points) indicating mask presence.
        """
        H_mask, W_mask = masks_np.shape[1], masks_np.shape[2]

        if orig_shape is not None:
            H_orig, W_orig = orig_shape
            # 缩放坐标到mask尺寸
            i = (i * H_mask / H_orig).astype(int)
            j = (j * W_mask / W_orig).astype(int)

        # 防止越界，裁剪坐标范围
        i_clipped = np.clip(i.reshape(-1), 0, H_mask - 1)
        j_clipped = np.clip(j.reshape(-1), 0, W_mask - 1)

        print(f"masks_np shape: {masks_np.shape}")
        print(f"max i: {i_clipped.max()}, max j: {j_clipped.max()}")
        print(f"min i: {i_clipped.min()}, min j: {j_clipped.min()}")

        # 取出对应位置的mask值，shape (N_masks, num_points)
        index = masks_np[:, i_clipped, j_clipped]

        return index

    def index_for_feature(self, index, clip_features):
        """
            return sample ray mean clip featrue (torch.float32)
        """

        if index.sum() == 0:
            return torch.zeros((512), device=self.device)

        # return torch.mean(clip_features[index], dim = 0)
        return clip_features[index][0]

    def indexs_for_feature(self, indexs, clip_features):
        """
            for large data
        """
        indexs = torch.from_numpy(indexs).float()
        result_tensor = torch.matmul(indexs, clip_features)

        return result_tensor

    def build_clip_index(self, masks_np, image):

        '''
            return : clip_features
        '''

        masks_num, H, W = masks_np.shape[0], masks_np.shape[1], masks_np.shape[2]
        image_np = image.detach().cpu().numpy()
        images_sam_np = image_np.copy()
        images_sam_np = np.tile(images_sam_np, (masks_num, 1, 1, 1))  # (n, H, W, 3)

        # images_sam_np[~masks_np] *= 0.3
        images_sam_np[~masks_np] = [0, 0, 0]
        images_sam_np *= 255

        return self.clip_image_feature(images_sam_np)

    def choose_pixels(H_mask, W_mask, way='random', num_samples=10000):
        """
        生成像素坐标，保证坐标在mask尺寸范围内。
        H_mask, W_mask: mask的高和宽
        way:
            - 'all': 返回mask内所有坐标
            - 'random': 随机采样num_samples个坐标
            - 'test', 'confirm': 方便测试用的单点坐标
        """
        if way == 'all':
            y_coords, x_coords = np.indices((H_mask, W_mask))
            y_coords = y_coords.reshape(-1)
            x_coords = x_coords.reshape(-1)
            num = H_mask * W_mask
        elif way == 'random':
            all_coords = np.indices((H_mask, W_mask), dtype=int)
            all_coords = np.stack(all_coords, axis=-1).reshape(-1, 2)
            # 随机采样
            random_sample_coords = all_coords[np.random.choice(all_coords.shape[0], size=num_samples, replace=False)]
            y_coords = random_sample_coords[:, 0]
            x_coords = random_sample_coords[:, 1]
            num = num_samples
        elif way == 'test':
            y_coords = np.array([213])
            x_coords = np.array([673])
            num = 1
        elif way == 'confirm':
            y_coords = np.array([279])
            x_coords = np.array([706])
            num = 1
        else:
            raise ValueError(f"Unknown way: {way}")

        return y_coords, x_coords, num

    def choose_masks_index(self, text_probs, y_coords, x_coords, masks_np, class_index=0):

        '''
            Find the point with the highest probability and return its masks_index.
        '''
        text_probs = text_probs.detach().cpu().numpy()

        index = np.nanargmax(text_probs[:, class_index])
        index_y = y_coords[index]
        index_x = x_coords[index]

        masks_index = masks_np[:, index_y, index_x]

        return masks_index

    def show_index_masks(self, index_masks, masks_np, image_np):

        mask = np.squeeze(masks_np[index_masks, :, :])

        image_np[mask] = (255, 0, 0)

        plt.imshow(image_np)

        return

    def get_masks_from_path(self, image_path):

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        """
            masks : (list):
                dict_keys([
                    'segmentation', ----- (mask) 
                    'area',         ----- (mask num)
                    'bbox',         ----- (XYWH)
                    'predicted_iou', ---- (score)
                    'point_coords', ----- (sample point)
                    'stability_score', -- (other socre)
                    'crop_box'       ---- (crop_image)
                    ])
        """

        masks = self.get_entire_image_mask(self.mask_generator, image_np)

        return masks

    def get_masks_from_tensor(self, image_tensor):

        image_np = (image_tensor.detach().cpu().numpy() * 255).astype(np.uint8)

        masks = self.get_entire_image_mask(self.mask_generator, image_np)

        return masks
