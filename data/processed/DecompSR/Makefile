DATA=../../raw/qa

DecompSR.jsonl: $(DATA)/clean/*_test_*.jsonl  $(DATA)/noise/*_test_*.jsonl
	cat $(DATA)/clean/*_test_*.jsonl > questions.jsonl
	cat $(DATA)/noise/*_test_*.jsonl >> questions.jsonl
	mv questions.jsonl DecompSR.jsonl

clean:
	rm -f DecompSR.jsonl
