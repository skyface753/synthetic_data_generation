
test-clean:
	rm -rf tmp

test-extract_cli:
	mkdir -p tmp/foregrounds
	python SynDataGenYOLO/cli/extract_cli.py --input_dir test_data/foregrounds --output_dir tmp/foregrounds --margin 10

test-copy-backgrounds:
	mkdir -p tmp/backgrounds
	cp -r test_data/backgrounds/* tmp/backgrounds/

test-data-generation:
	python SynDataGenYOLO/cli/data_generation_cli.py -in_dir tmp -out_dir tmp/out -img_number 5 --distractor_objects Mud --output_mode YOLO --overwrite_output

show-tmp-with-bboxes:
	python SynDataGenYOLO/cli/show_images_with_bboxes_cli.py --input tmp/out

test-data-generation-with-bboxes:
	mkdir -p tmp/labels
	mkdir -p tmp/backgrounds
	cp -r test_data/backgrounds_with_bboxes/images/* tmp/backgrounds/
	cp -r test_data/backgrounds_with_bboxes/labels/* tmp/labels/
	python SynDataGenYOLO/cli/data_generation_cli.py -in_dir tmp -out_dir tmp/out -img_number 5 \
		--distractor_objects Mud --output_mode YOLO --yolo_input --fixed_image_sizes --overwrite_output

test-with-yolo-backgrounds: test-clean test-extract_cli test-data-generation-with-bboxes show-tmp-with-bboxes



test-all: test-clean test-extract_cli test-copy-backgrounds test-data-generation show-tmp-with-bboxes