
test-clean:
	rm -rf tmp

test-extract_cli:
	mkdir -p tmp/foregrounds
	python synthetic_data_gen/cli/extract_cli.py --input_dir test_data/foregrounds --output_dir tmp/foregrounds --margin 10

test-copy-backgrounds:
	mkdir -p tmp/backgrounds
	cp -r test_data/backgrounds/* tmp/backgrounds/

test-data-generation:
	python synthetic_data_gen/cli/data_generation_cli.py -in_dir tmp -out_dir tmp/out -img_number 5 --distractor_objects Mud --yolo_output


test-all: test-clean test-extract_cli test-copy-backgrounds test-data-generation