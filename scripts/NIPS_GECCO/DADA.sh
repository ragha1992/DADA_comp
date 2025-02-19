python -u run.py --metric affiliation --t $(seq 0.115 0.001 0.125) --norm 1 --use_gpu True --gpu 0 --root_path ./dataset/evaluation_dataset --data GECCO --model ./DADA --des 'zero_shot' --batch_size 32
python -u run.py --metric auc r_auc vus --norm 1 --use_gpu True --gpu 0 --root_path ./dataset/evaluation_dataset --data GECCO --model ./DADA --des 'zero_shot' --batch_size 32
python -u run.py --metric best_f1 --norm 1 --use_gpu True --gpu 0 --root_path ./dataset/evaluation_dataset --data GECCO --model ./DADA --des 'zero_shot' --batch_size 32
  






