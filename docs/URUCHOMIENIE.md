# LCF3D na nuScenes — trening, walidacja i testy

Repozytorium zawiera lokalnie rozwijany kod LCF3D, zmodyfikowane MMDetection3D
1.4.0, konfiguracje trzech modeli i narzędzia przygotowania danych oraz oceny.
Oryginalny projekt: CarloSgaravatti/LCF3D, commit bazowy `e098ded`.
Zachowano licencje i dokumentację autorów. Skrypty w `scripts/` są punktami
wejścia dla tej wersji; starsze konfiguracje `*_local.json` dokumentują lokalne
eksperymenty. Przenośna konfiguracja to `src/configs/lcf3d_nuscenes.json`.

## Instalacja

Środowisko źródłowe: Linux, Python 3.10.20, PyTorch 2.7.1+cu128,
torchvision 0.22.1+cu128, MMCV 2.1.0, MMDetection 3.2.0, MMEngine 0.10.7.
Lista zależności: `requirements.txt`; testy: `requirements-dev.txt`.
`environment-packages.txt` zapisuje wersje z istniejącego środowiska (również
pakiety innych eksperymentów); nie zastępuje instrukcji instalacji.

```bash
gh repo clone mmr0z/LCF3D-nuscenes
cd LCF3D-nuscenes
conda create -n lcf3d-repo python=3.10 -y
conda activate lcf3d-repo
# Wymagane: kompilator C++, CUDA Toolkit 12.8 / nvcc i zgodny sterownik NVIDIA.
# Ustaw CUDA_HOME na katalog z zainstalowanym Toolkit, jeśli nie jest wykrywany.
bash scripts/install.sh
bash scripts/download_models.sh
```

Instalator kompiluje operatory CUDA MMCV. Zapisano wersje z lokalnego środowiska;
pełnej instalacji w czystym środowisku nie zweryfikowano. Kompilacja może wymagać
ustawienia `TORCH_CUDA_ARCH_LIST` odpowiednio do GPU. Nie instaluj równolegle
`mmcv-lite`. Wersja MMDetection3D z tego repozytorium zawiera kod Frustum Localizer
oraz zmienione metryki i musi być zainstalowana przez `pip install -e`.

## Modele

W prywatnym wydaniu `models-v1` są dokładne pliki wag wskazane przez lokalną
konfigurację pełnej fuzji. `scripts/download_models.sh` pobiera je przez zalogowane
`gh` i sprawdza SHA-256 z `docs/models.sha256`.

| Plik w checkpoints/ | Model | Źródłowy checkpoint |
|---|---|---|
| camera.pth | Faster R-CNN, kamera | best_coco_bbox_mAP_iter_85134.pth |
| centerpoint.pth | CenterPoint, LiDAR | epoch_20.pth |
| frustum.pth | Frustum PointNet Localizer | best_IoU_3D_epoch_98.pth |

Kod modeli znajduje się w `src/mmdetection3d/mmdet3d/models/` (m.in. detektory
frustum), konfiguracje w `src/model_configs/nuscenes/`, kod fuzji w
`src/base_inference.py` i `src/multi_view_inference.py`. Gałąź 2D korzysta z
MMDetection. Bieżąca konfiguracja zachowuje ustawienia eksperymentu:
odzyskiwanie detekcji Frustum jest ograniczone do klasy `traffic_cone`.

## Dane

nuScenes nie jest dołączony do repozytorium. Potrzebujesz pobranych danych
`v1.0-trainval`, `samples`, `sweeps`, `maps` oraz przygotowanych adnotacji
MMDetection3D: `nuscenes_infos_train.pkl`, `nuscenes_infos_val.pkl`,
`nuscenes_dbinfos_train.pkl` i bazy obiektów dla augmentacji treningu LiDAR.
Instrukcje konwersji są w `src/mmdetection3d/docs/en/advanced_guides/datasets/`.
Wszystkie poniższe komendy wykonuj z katalogu głównego repozytorium.

```bash
export LCF3D_NUSCENES_ROOT=/sciezka/do/nuscenes
bash src/prepare_nuscenes_camera2d.sh
# Ta sama ścieżka dla przygotowania, treningu i walidacji Frustum:
export LCF3D_FRUSTUM_ROOT="$PWD/data/frustum_nuscenes"
bash src/prepare_nuscenes_frustum.sh
```

Opcjonalny podzbiór, zgodny z domyślną ścieżką skryptu treningowego:

```bash
python src/dataset_utils/subsample_frustum_dataset.py \
  --source-dir data/frustum_nuscenes --output-dir data/frustum_nuscenes_compact
export LCF3D_FRUSTUM_ROOT="$PWD/data/frustum_nuscenes_compact"
```

Ograniczona walidacja Frustum służy do eksperymentów; wyniki końcowe fuzji
raportuj na całym splicie nuScenes `val`.

## Trening

```bash
bash src/train_nuscenes_clean.sh       # CenterPoint
bash src/train_nuscenes_camera2d.sh    # Faster R-CNN
bash src/train_nuscenes_frustum.sh     # Frustum Localizer
```

Trening zapisuje checkpointy i logi w `work_dirs/`; konfiguracja Frustum ma
100 epok i walidację co 2 epoki. Treningi mogą trwać długo i wymagają GPU.
Skrypt Frustum domyślnie używa batch size 64; zmień go przez
`LCF3D_FRUSTUM_BATCH_SIZE`. Po własnym treningu wskaż nowe checkpointy
w kopii konfiguracji fuzji i ustaw `LCF3D_FUSION_CFG` na jej ścieżkę.
Przykład wznowienia Frustum:

```bash
bash src/train_nuscenes_frustum.sh resume=True
```

Końcowe argumenty tego skryptu są opcjami konfiguracji MMEngine (`key=value`).

## Walidacja i testowanie modelu

```bash
bash scripts/validate_camera.sh       # COCO AP/AR gałęzi 2D
bash scripts/validate_frustum.sh      # IoU lokalizatora
bash scripts/validate_fusion.sh       # pełny val: nuScenes mAP, NDS, błędy TP
# Krótki test inferencji, bez oficjalnych metryk dla niepełnego splitu:
bash scripts/validate_fusion.sh --max_samples 5
```

Dla własnego lokalizatora ustaw `LCF3D_FRUSTUM_CHECKPOINT`; dla gałęzi 2D
`LCF3D_CAMERA_CHECKPOINT`. Pełna fuzja czyta wszystkie trzy wagi z JSON.
Wyniki i raporty CSV/Markdown trafiają do `work_dirs/`.
Skrypty oceny domyślnie używają oznaczonego splitu `val`. Nie są to wyniki
ukrytego testu nuScenes; jego oficjalna ocena wymaga osobnego zgłoszenia
predykcji do serwera benchmarku.

## Testy kodu

```bash
bash scripts/test.sh -q
```

Testy sprawdzają kolejność klas nuScenes, agregację widoków oraz analizę metryk.
Nie zastępują pełnego treningu ani walidacji na GPU. Dane, logi i checkpointy
są wyłączone z Git; wagi są dystrybuowane przez prywatne wydanie.
