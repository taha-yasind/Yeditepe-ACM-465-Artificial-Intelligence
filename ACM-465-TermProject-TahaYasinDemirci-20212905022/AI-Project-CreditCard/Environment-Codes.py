# mevcut tüm environment'ları listele
conda env list

# base environment'ı yaml dosyasına aktar
conda env export > environment.yaml

# yeni environment oluştur
conda create -n creditcard_env

# environment'ı aktif et
conda activate creditcard_env

# aktif environment'ı deaktif et (base'e dön)
conda deactivate

# yaml dosyasından environment kur
conda env update -n creditcard_env -f environment.yaml

# environment'ı sil
conda env remove -n creditcard_env

# aktif environment'daki paketleri listele
conda list

# tüm paketleri güncelle
conda upgrade --all

# belirli bir paket kur
conda install numpy