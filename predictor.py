#  https://github.com/isl-org/ZoeDept
# ilk donanımı seçiyoruz --> self._get_device()
# modeli initialize ediyoruz --> self.model = self._initialize_model(model_repo="isl-org/ZoeDepth", model_name="ZoeD_N").to(self.device)
# fotoğrafı kaydetmek için kullanacağımız fonksiyonu seçtik --> save_colored_depth(self, depth_numpy, output_path)
# inference key i yapcaz hesaplamaların yapıldığı kısım. --> calculate_depthmap(self,image_path,output_path):

import torch
from PIL import Image

from misc import colorize


class DepthEstimationModel:
    def __init__(self) -> None:
        self.device = self._get_device()
        self.model = self._initialize_model(
            model_repo="isl-org/ZoeDepth", model_name="ZoeD_N"
        ).to(self.device)
        """
        .to(device) ifadesi PyTorch'a özgü bir özelliktir. Bu özellik, bir PyTorch tensorünü belirli bir cihaza taşımak için kullanılır. Bu cihaz genellikle bir GPU ("cuda") veya CPU ("cpu") olabilir.
        yani bu şekilde modeli hangi donanımı seçtiysek ona göndermiş oluyoruz.
        """

    def _get_device(self):
        """
        CPU GPU seçimi yapmamız gerekiyor, bir yapay zeka modeli kullanıyoruz, bunu seçmek gerekiyor.
        eğer cpu kurulumu yaptıysak cpu seçmeyi sağlicak gpu kurulumu yaptıysak gpu seçmemizi sağlicak bu nedenle
        """
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _initialize_model(self, model_repo="isl-org/ZoeDepth", model_name="ZoeD_N"):
        """
        Burda intelin yayınladığı bir modeli kullanıcaz
        https://github.com/isl-org/ZoeDepth

        """
        torch.hub.help(
            "intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True
        )  # -- githubda böyle yap demiş
        model = torch.hub.load(
            model_repo, model_name, pretrained=True, skip_validation=False
        )  # -- modeli çekiyoruz bunlar github sayfasında yazıyor
        model.eval()  #  inferenc model kullanacağımız için evaluation moda a sokuyoruz

        print("Model initialized.")
        return model

    def save_colored_depth(self, depth_numpy, output_path):
        colored = colorize(
            depth_numpy
        )  # en son renklendirme yapıyor misc den çıktik githubda var
        Image.fromarray(colored).save(output_path)  # pillow ile kaydettik
        print("Image saved.")

    def calculate_depthmap(self, image_path, output_path):
        """
        resimi alcak hesaplamaları yapcak kaydetcek
        """
        image = Image.open(image_path).convert("RGB")
        print("Image read.")
        depth_numpy = self.model.infer_pil(image)  # hesaplama işlemi burda yapılıyor.
        self.save_colored_depth(
            depth_numpy, output_path
        )  # yukardaki fonk yardımıyla renklendirme işlemi yapılıp kaydediyoruz.
        return f"Image saved to {output_path}"


# model = DepthEstimationModel()
# model.calculate_depthmap("./test_image.png", "output_image.png")
