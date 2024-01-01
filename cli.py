"""
Burda modelimizi komut satırından çalıştırmak için gerekli  command line interface hazırlicaz
dışarıdan yazacağımız input image path ve outhput pathleri tanımlıyoruz
"""

import argparse

from predictor import DepthEstimationModel


def main():
    parser = argparse.ArgumentParser(description="Depth Estimation using ZoeDepth.")
    parser.add_argument(
        "input_image", help="Path to input image."
    )  # açıklamasını helpini yazıyoruz
    parser.add_argument("output_image", help="Path to output depth map.")
    args = parser.parse_args()

    model = DepthEstimationModel()
    result = model.calculate_depthmap(args.input_image, args.output_image)
    print(result)


# python cli.py --help deyincede yazdıklarımız help menüsünde güzel bir şekilde görünür.

# kullanımı

# python cli.py test_image.png output_cli.png


# burdaki main fonksiyonu sadece python cli.py deyince otomatik çalışcak biryerde import edilirse otomatik çalışmicak.
if __name__ == "__main__":
    main()
