#./版本二/dev.txt
nohup python dense_inference.py test1.txt>/dev/null 2>&1 & nohup python ner_inference.py test1.txt>/dev/null 2>&1 
wait 
python aggregate_result.py
