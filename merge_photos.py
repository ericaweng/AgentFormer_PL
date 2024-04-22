import os
from PIL import Image


def concatenate_images(args):
    directory = args.directory
    images = [Image.open(os.path.join(directory, img)) for img in os.listdir(directory) if
              img.endswith(('.png', '.jpg', '.jpeg'))]
    # images.sort()  # Sorting to maintain consistent order

    i = 0
    while i < len(images) - 1:
        imgs = images[i:i + args.images_per_concatenation]

        # Assuming the images are 3:4 in aspect ratio
        desired_width = imgs[0].width
        desired_height = imgs[0].height

        # Concatenating along the long edge
        final_img = Image.new('RGB', (desired_width, args.images_per_concatenation * desired_height))
        for im_i in range(args.images_per_concatenation):
            final_img.paste(imgs[im_i], (0, desired_height * im_i))

        # add white space to adjust photo to either 2:3 or 3:2 aspect ratio, depending on which adds less whitespace
        current_aspect_ratio = final_img.width / final_img.height
        if (current_aspect_ratio - 2/3)**2 < (current_aspect_ratio - 3/2)**2:
            final_aspect_ratio = 2 / 3
        else:
            final_aspect_ratio = 3 / 2

        if current_aspect_ratio < final_aspect_ratio:
            # Add white space on the right
            new_width = int(final_img.height * final_aspect_ratio)
            result_img = Image.new('RGB', (new_width, final_img.height), 'white')
            result_img.paste(final_img, (0, 0))
        elif current_aspect_ratio > final_aspect_ratio:
            # Add white space at the bottom
            new_height = int(final_img.width / final_aspect_ratio)
            result_img = Image.new('RGB', (final_img.width, new_height), 'white')
            result_img.paste(final_img, (0, 0))
        else:
            result_img = final_img

        # Save the result image
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)
        result_img.save(os.path.join(f'{args.output_directory}/concat_{i // 2}.jpg'))

        i += args.images_per_concatenation


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Concatenate images')
    parser.add_argument('--directory', '-d',type=str, help='Directory containing images', default='../Photos-001')
    parser.add_argument('--output_directory', '-od',type=str, help='Directory containing images', default='../viz/Photos-001')
    parser.add_argument('--images_per_concatenation','-i', type=int, default=2, help='Number of images to concatenate')
    args = parser.parse_args()

    concatenate_images(args)
