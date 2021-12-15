# tf_v1_object_detection
tf_v1_object_detection


Base repo https://github.com/tensorflow/models/tree/master/research/object_detection


# Windows installation notes

TF GPU INSTALLATION CODE : conda install -c anaconda tensorflow-gpu==1.15.0

conda install -c anaconda protobuf

git clone https://github.com/tensorflow/models.git

cd models\research\

C:\tf_v1_object_detection\protoc\bin\protoc.exe object_detection/protos/ssd.proto --python_out=.

cd object_detection\packages\tf1
copy setup.py C:\tf_v1_object_detection\models\research\.

cd ..
cd ..
cd ..

python -m pip install --use-feature=2020-resolver .

SET PYTHONPATH=C:\tf_v1_object_detection\models\research;C:\tf_v1_object_detection\models\research\slim

python object_detection/builders/model_builder_tf1_test.py



# Training Notes

sahinkul_presence_absence_check

1
[COLAB]!python3 data_augmentation.py
[COLAB]!python3 generate_tfrecord.py

2
[COLAB]!python3 xmltocsv.py
[COLAB]!python3 generate_tfrecord.py

[COLAB]!python3 legacy/train.py --logtostderr --train_dir=training/sahinkul_presence_absence_check --pipeline_config_path=training/sahinkul_presence_absence_check/faster_rcnn_inception_v2_coco.config


[LOCAL]python export_inference_graph.py --input_type image_tensor --pipeline_config_path=training/sahinkul_presence_absence_check/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix=training/sahinkul_presence_absence_check/model.ckpt-22243 --output_directory=trained_models/sahinkul_presence_absence_check

[LOCAL]python sahinkul_presence_absence_check.py

