import os
from PIL import Image


def concatenate_images_vertically(args):
    directory = args.directory
    images = [Image.open(os.path.join(directory, img)) for img in os.listdir(directory) if
              img.endswith(('.png', '.jpg', '.jpeg'))]

    i = 0

    while i < len(images) - 1:
        imgs = images[i:i + args.images_per_concatenation]

        # Assuming the images are 3:4 in aspect ratio
        desired_width = imgs[0].width
        desired_height = int(round(imgs[0].width * 3/2)) #imgs[0].height

        # Concatenating along the short edge
        final_img = Image.new('RGB', (desired_width, desired_height), 'white')
        current_y_axis = 0
        for im_i in range(args.images_per_concatenation):
            if im_i >= len(imgs):
                break
            img = imgs[im_i]
            # make images the same height
            if img.width != desired_width:
                img = img.resize((desired_width, int(desired_width * img.height / img.width)))
            final_img.paste(img, (0, current_y_axis))
            current_y_axis += img.height
        # crop off the first 1/3 of the image from the left
        # final_img = final_img.crop((int(round(desired_width/3)), 0, desired_width, desired_height))

        # Save the result image
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

        save_path = os.path.join(f'{args.output_directory}/{args.directory}_{i // 2}.jpg')
        final_img.save(save_path)

        i += args.images_per_concatenation

def concatenate_images_horizontal_crop(args):
    directory = args.directory
    images = [Image.open(os.path.join(directory, img)) for img in os.listdir(directory) if
              img.endswith(('.png', '.jpg', '.jpeg'))]
    # images.sort()  # Sorting to maintain consistent order

    i = 0

    while i < len(images) - 1:
        imgs = images[i:i + args.images_per_concatenation]

        desired_height = int(imgs[0].height * 2/3-100)

        # Concatenating along the long edge
        final_img = Image.new('RGB', (int(round(desired_height*3/2)), desired_height), 'white')
        current_x_axis = 0
        for im_i in range(args.images_per_concatenation):
            if im_i >= len(imgs):
                break
            img = imgs[im_i]
            final_img.paste(img, (current_x_axis, 0))
            current_x_axis += img.width

        # remove middle chunk of image and paste two sides together
        # final_img = final_img.crop((0, 0, int(round(desired_height*3/2)), desired_height))

        # Save the result image
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

        save_path = os.path.join(f'{args.output_directory}/{args.directory}_{i // 2}.jpg')
        final_img.save(save_path)

        i += args.images_per_concatenation


def concatenate_images(args):
    directory = args.directory
    images = [Image.open(os.path.join(directory, img)) for img in os.listdir(directory) if
              img.endswith(('.png', '.jpg', '.jpeg'))]
    # images.sort()  # Sorting to maintain consistent order

    i = 0

    while i < len(images) - 1:
        imgs = images[i:i + args.images_per_concatenation]

        # Assuming the images are 3:4 in aspect ratio
        # desired_width = imgs[0].width
        # desired_height = int(round(imgs[0].width * 3/2)) #imgs[0].height
        desired_height = imgs[0].height
        desired_width = int(round(imgs[0].height * 3/2)) #imgs[0].height

        # *args.images_per_concatenation
        # Concatenating along the long edge
        # final_img = Image.new('RGB', (desired_width, args.images_per_concatenation * desired_height), 'white')
        final_img = Image.new('RGB', (desired_width, desired_height), 'white')
        current_x_axis = 0
        for im_i in range(args.images_per_concatenation):
            if im_i >= len(imgs):
                break
            img = imgs[im_i]
            print(f"{img.width} {img.height}")
            # make images the same width
            if img.height != desired_height:
                print(f"{desired_height=}")
            # if img.width != desired_width:
                img = img.resize((int(desired_height * img.width / img.height), desired_height))
                # img = img.resize((desired_width, int(desired_width * img.height / img.width)))
                print(f"after {img.width} {img.height}")
            # make images the same height
            # final_img.paste(img, (0, desired_height * im_i))
            final_img.paste(img, (current_x_axis, 0))
            current_x_axis += img.width
            # print(f"{current_x_axis=}")


        # add white space to adjust photo to either 2:3 or 3:2 aspect ratio, depending on which adds less whitespace
        # current_aspect_ratio = final_img.width / final_img.height
        # if (current_aspect_ratio - 2/3)**2 < (current_aspect_ratio - 3/2)**2:
        #     final_aspect_ratio = 2 / 3
        # else:
        #     final_aspect_ratio = 3 / 2

        # if current_aspect_ratio < final_aspect_ratio:
        #     # Add white space on the right
        #     new_width = int(final_img.height * final_aspect_ratio)
        #     result_img = Image.new('RGB', (new_width, final_img.height), 'white')
        #     result_img.paste(final_img, (0, 0))
        # elif current_aspect_ratio > final_aspect_ratio:
        #     # Add white space at the bottom
        #     new_height = int(final_img.width / final_aspect_ratio)
        #     result_img = Image.new('RGB', (final_img.width, new_height), 'white')
        #     result_img.paste(final_img, (0, 0))
        # else:
        #     result_img = final_img

        # Save the result image
        if not os.path.exists(args.output_directory):
            os.makedirs(args.output_directory)

        save_path = os.path.join(f'{args.output_directory}/{args.directory}_{i // 2}.jpg')
        final_img.save(save_path)

        i += args.images_per_concatenation


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Concatenate images')
    parser.add_argument('--directory', '-d',type=str, help='Directory containing images', default='phone-horiz')
    parser.add_argument('--output_directory', '-od',type=str, help='Directory containing images', default='../viz/concatted_photos')
    parser.add_argument('--images_per_concatenation','-i', type=int, default=2, help='Number of images to concatenate')
    parser.add_argument('--horiz','-hz', action='store_true', help='Concatenate horizontally')
    parser.add_argument('--horiz_crop','-hc', action='store_true', help='Concatenate horizontally')

    args = parser.parse_args()

    if args.horiz:
        concatenate_images(args)
    elif args.horiz_crop:
        concatenate_images_horizontal_crop(args)
    else:
        concatenate_images_vertically(args)
