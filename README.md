1.下載好數據集，預設使用的是花分類數據集，下載地址: https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz

2.在`train.py中將`--data-path`設置成解壓縮後的`flower_photos`文件夾絕對路徑

3.下載預訓練權重，在`model.py`文件中每個模型都有提供預訓練權重的下載地址，根據使用的模型下載對應的預訓練權重

4.在`train.py`中將`--weights`參數設置成下載好的預訓練權重路徑

5.設置好數據集的路徑`--data-path`以及預訓練權重的路徑`--weights`就能使用`train.py`開始訓練(訓練過程中自動生成`class_indices.json`文件)

6.在`predict.py`中導入和訓練中同樣的模型，並將`model_weight_path`設置成訓練好的模型權重路徑(預設儲存在weights文件夾下)

7.在`predict.py`中將`img_path`設置成需要預測的圖片絕對路徑

8.設置好權重路徑`model_weight_path`和預測的圖片路徑`img_path`就能使用`predict.py`進行預測

9.如果要使用自己的數據，可依照花分類數據的文件結構進行擺放(即一個類別對應一個文件夾)，並將訓練以及預測中的`num_classes`設置成自己數據的類別數量
