python main.py --model-path ./../falcon --save-name falcon_std --attack gcg --batch-size 128
python main.py --model-path ./../falcon --save-name falcon_icd_2 --attack gcg --batch-size 128 --defense icd --icd-num 2

python main.py --model-path ./../falcon --save-name falcon_remind --attack gcg --batch-size 128 --defense remind
python main.py --model-path ./../falcon --save-name falcon_icd_1 --attack gcg --batch-size 128 --defense icd --icd-num 1

python main.py --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_std --attack autodan --batch-size 32
python main.py --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_icd_1 --attack autodan --batch-size 32 --defense icd --icd-num 1
python main.py --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_icd_2 --attack autodan --batch-size 32 --defense icd --icd-num 2
python main.py --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_remind --attack autodan --batch-size 32 --defense remind

python main.py --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_std --attack autodan --batch-size 64
python main.py --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_icd_1 --attack autodan --batch-size 64 --defense icd --icd-num 1
python main.py --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_icd_2 --attack autodan --batch-size 64 --defense icd --icd-num 2
python main.py --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_remind --attack autodan --batch-size 64 --defense remind

python main.py --model-path ./../falcon --save-name falcon_auto_std --attack autodan --batch-size 32
python main.py --model-path ./../falcon --save-name falcon_auto_icd_1 --attack autodan --batch-size 32 --defense icd --icd-num 1
python main.py --model-path ./../falcon --save-name falcon_auto_icd_2 --attack autodan --batch-size 32 --defense icd --icd-num 2
python main.py --model-path ./../falcon --save-name falcon_auto_remind --attack autodan --batch-size 32 --defense remind


# GLUE

python glue.py --model-path /data/models/vicuna-7b-v1.5 --save-name vicuna_std
python glue.py --model-path /data/models/vicuna-7b-v1.5 --save-name vicuna_icd_1 --defense icd --icd-num 1
python glue.py --model-path /data/models/vicuna-7b-v1.5 --save-name vicuna_icd_2 --defense icd --icd-num 2
python glue.py --model-path /data/models/vicuna-7b-v1.5 --save-name vicuna_remind --defense remind


python glue.py --model-path /data/models/Llama-2-7b-chat-hf --save-name llama_std
python glue.py --model-path /data/models/Llama-2-7b-chat-hf --save-name llama_icd_1 --defense icd --icd-num 1
python glue.py --model-path /data/models/Llama-2-7b-chat-hf --save-name llama_icd_2 --defense icd --icd-num 2
python glue.py --model-path /data/models/Llama-2-7b-chat-hf --save-name llama_remind --defense remind


python glue.py --model-path /data/models/falcon --save-name falcon_std
python glue.py --model-path /data/models/falcon --save-name falcon_icd_1 --defense icd --icd-num 1
python glue.py --model-path /data/models/falcon --save-name falcon_icd_2 --defense icd --icd-num 2
python glue.py --model-path /data/models/falcon --save-name falcon_remind --defense remind

# Self check
# single prompt
python self_check.py --attack gcg --model-path ./../vicuna-7b-v1.5 --save-name vicuna_cac_1 --check-round 1 --fname ../result_AdvLLM/vicuna_std.json
python self_check.py --attack gcg --model-path ./../vicuna-7b-v1.5 --save-name vicuna_cac_2 --check-round 2 --fname ../result_AdvLLM/vicuna_std.json
python self_check.py --attack gcg --model-path ./../vicuna-7b-v1.5 --save-name vicuna_cac_3 --check-round 3 --fname ../result_AdvLLM/vicuna_std.json

python self_check.py --attack autodan --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_cac_1 --check-round 1 --fname ../result_AdvLLM/vicuna_auto_std.json
python self_check.py --attack autodan --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_cac_2 --check-round 2 --fname ../result_AdvLLM/vicuna_auto_std.json
python self_check.py --attack autodan --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_cac_3 --check-round 3 --fname ../result_AdvLLM/vicuna_auto_std.json

python self_check.py --attack gcg --model-path ./../Llama-2-7b-chat-hf --save-name llama_cac_1 --check-round 1 --fname ../result_AdvLLM/llama_std.json
python self_check.py --attack gcg --model-path ./../Llama-2-7b-chat-hf --save-name llama_cac_2 --check-round 2 --fname ../result_AdvLLM/llama_std.json
python self_check.py --attack gcg --model-path ./../Llama-2-7b-chat-hf --save-name llama_cac_3 --check-round 3 --fname ../result_AdvLLM/llama_std.json

python self_check.py --attack autodan --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_cac_1 --check-round 1 --fname ../result_AdvLLM/llama_auto_std.json
python self_check.py --attack autodan --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_cac_2 --check-round 2 --fname ../result_AdvLLM/llama_auto_std.json
python self_check.py --attack autodan --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_cac_3 --check-round 3 --fname ../result_AdvLLM/llama_auto_std.json


python self_check.py --attack gcg --model-path ./../falcon --save-name falcon_cac_1 --check-round 1 --fname ../result_AdvLLM/falcon_std.json
python self_check.py --attack gcg --model-path ./../falcon --save-name falcon_cac_2 --check-round 2 --fname ../result_AdvLLM/falcon_std.json
python self_check.py --attack gcg --model-path ./../falcon --save-name falcon_cac_3 --check-round 3 --fname ../result_AdvLLM/falcon_std.json


python self_check.py --attack autodan --model-path ./../falcon --save-name falcon_auto_cac_1 --check-round 1 --fname ../result_AdvLLM/falcon_auto_std.json
python self_check.py --attack autodan --model-path ./../falcon --save-name falcon_auto_cac_2 --check-round 2 --fname ../result_AdvLLM/falcon_auto_std.json
python self_check.py --attack autodan --model-path ./../falcon --save-name falcon_auto_cac_3 --check-round 3 --fname ../result_AdvLLM/falcon_auto_std.json


# history backup
python self_check.py --attack gcg --model-path ./../vicuna-7b-v1.5 --save-name vicuna_cac_backup_1 --check-round 1 --fname ../result_AdvLLM/vicuna_std.json --backup
python self_check.py --attack gcg --model-path ./../vicuna-7b-v1.5 --save-name vicuna_cac_backup_2 --check-round 2 --fname ../result_AdvLLM/vicuna_std.json --backup
python self_check.py --attack gcg --model-path ./../vicuna-7b-v1.5 --save-name vicuna_cac_backup_3 --check-round 3 --fname ../result_AdvLLM/vicuna_std.json --backup

python self_check.py --attack autodan --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_cac_backup_1 --check-round 1 --fname ../result_AdvLLM/vicuna_auto_std.json --backup
python self_check.py --attack autodan --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_cac_backup_2 --check-round 2 --fname ../result_AdvLLM/vicuna_auto_std.json --backup
python self_check.py --attack autodan --model-path ./../vicuna-7b-v1.5 --save-name vicuna_auto_cac_backup_3 --check-round 3 --fname ../result_AdvLLM/vicuna_auto_std.json --backup

python self_check.py --attack gcg --model-path ./../Llama-2-7b-chat-hf --save-name llama_cac_backup_1 --check-round 1 --fname ../result_AdvLLM/llama_std.json --backup
python self_check.py --attack gcg --model-path ./../Llama-2-7b-chat-hf --save-name llama_cac_backup_2 --check-round 2 --fname ../result_AdvLLM/llama_std.json --backup
python self_check.py --attack gcg --model-path ./../Llama-2-7b-chat-hf --save-name llama_cac_backup_3 --check-round 3 --fname ../result_AdvLLM/llama_std.json --backup

python self_check.py --attack autodan --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_cac_backup_1 --check-round 1 --fname ../result_AdvLLM/llama_auto_std.json --backup
python self_check.py --attack autodan --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_cac_backup_2 --check-round 2 --fname ../result_AdvLLM/llama_auto_std.json --backup
python self_check.py --attack autodan --model-path ./../Llama-2-7b-chat-hf --save-name llama_auto_cac_backup_3 --check-round 3 --fname ../result_AdvLLM/llama_auto_std.json --backup


python self_check.py --attack gcg --model-path ./../falcon --save-name falcon_cac_backup_1 --check-round 1 --fname ../result_AdvLLM/falcon_std.json --backup
python self_check.py --attack gcg --model-path ./../falcon --save-name falcon_cac_backup_2 --check-round 2 --fname ../result_AdvLLM/falcon_std.json --backup
python self_check.py --attack gcg --model-path ./../falcon --save-name falcon_cac_backup_3 --check-round 3 --fname ../result_AdvLLM/falcon_std.json --backup


python self_check.py --attack autodan --model-path ./../falcon --save-name falcon_auto_cac_backup_1 --check-round 1 --fname ../result_AdvLLM/falcon_auto_std.json --backup
python self_check.py --attack autodan --model-path ./../falcon --save-name falcon_auto_cac_backup_2 --check-round 2 --fname ../result_AdvLLM/falcon_auto_std.json --backup
python self_check.py --attack autodan --model-path ./../falcon --save-name falcon_auto_cac_backup_3 --check-round 3 --fname ../result_AdvLLM/falcon_auto_std.json --backup


# Multi GCG
python main.py --attack suffix --suffix vicuna --model-path ./../vicuna-7b-v1.5 --save-name vicuna_V_std 
python main.py --attack suffix --suffix vicuna --model-path ./../vicuna-7b-v1.5 --save-name vicuna_V_icd_1 --defense icd --icd-num 1
python main.py --attack suffix --suffix vicuna --model-path ./../vicuna-7b-v1.5 --save-name vicuna_V_icd_2 --defense icd --icd-num 2
python main.py --attack suffix --suffix vicuna --model-path ./../vicuna-7b-v1.5 --save-name vicuna_V_remind --defense remind
python main.py --attack suffix --suffix llama --model-path ./../vicuna-7b-v1.5 --save-name vicuna_L_std 
python main.py --attack suffix --suffix llama --model-path ./../vicuna-7b-v1.5 --save-name vicuna_L_icd_1 --defense icd --icd-num 1
python main.py --attack suffix --suffix llama --model-path ./../vicuna-7b-v1.5 --save-name vicuna_L_icd_2 --defense icd --icd-num 2
python main.py --attack suffix --suffix llama --model-path ./../vicuna-7b-v1.5 --save-name vicuna_L_remind --defense remind

python main.py --attack suffix --suffix vicuna --model-path ./../Llama-2-7b-chat-hf --save-name llama_V_std 
python main.py --attack suffix --suffix vicuna --model-path ./../Llama-2-7b-chat-hf --save-name llama_V_icd_1 --defense icd --icd-num 1
python main.py --attack suffix --suffix vicuna --model-path ./../Llama-2-7b-chat-hf --save-name llama_V_icd_2 --defense icd --icd-num 2
python main.py --attack suffix --suffix vicuna --model-path ./../Llama-2-7b-chat-hf --save-name llama_V_remind --defense remind
python main.py --attack suffix --suffix llama --model-path ./../Llama-2-7b-chat-hf --save-name llama_L_std 
python main.py --attack suffix --suffix llama --model-path ./../Llama-2-7b-chat-hf --save-name llama_L_icd_1 --defense icd --icd-num 1
python main.py --attack suffix --suffix llama --model-path ./../Llama-2-7b-chat-hf --save-name llama_L_icd_2 --defense icd --icd-num 2
python main.py --attack suffix --suffix llama --model-path ./../Llama-2-7b-chat-hf --save-name llama_L_remind --defense remind

python main.py --attack suffix --suffix vicuna --model-path ./../falcon --save-name falcon_V_std 
python main.py --attack suffix --suffix vicuna --model-path ./../falcon --save-name falcon_V_icd_1 --defense icd --icd-num 1
python main.py --attack suffix --suffix vicuna --model-path ./../falcon --save-name falcon_V_icd_2 --defense icd --icd-num 2
python main.py --attack suffix --suffix vicuna --model-path ./../falcon --save-name falcon_V_remind --defense remind
python main.py --attack suffix --suffix llama --model-path ./../falcon --save-name falcon_L_std 
python main.py --attack suffix --suffix llama --model-path ./../falcon --save-name falcon_L_icd_1 --defense icd --icd-num 1
python main.py --attack suffix --suffix llama --model-path ./../falcon --save-name falcon_L_icd_2 --defense icd --icd-num 2
python main.py --attack suffix --suffix llama --model-path ./../falcon --save-name falcon_L_remind --defense remind


# ICA
python main.py --attack ica --ica-num 1 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_1
python main.py --attack ica --ica-num 2 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_2
python main.py --attack ica --ica-num 3 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_3
python main.py --attack ica --ica-num 4 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_4
python main.py --attack ica --ica-num 5 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_5
python main.py --attack ica --ica-num 6 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_6
python main.py --attack ica --ica-num 7 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_7
python main.py --attack ica --ica-num 8 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_8
python main.py --attack ica --ica-num 9 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_9
python main.py --attack ica --ica-num 10 --model-path ./../vicuna-7b-v1.5 --save-name vicuna_ICA_10

python main.py --attack ica --ica-num 1 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_1
python main.py --attack ica --ica-num 2 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_2
python main.py --attack ica --ica-num 3 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_3
python main.py --attack ica --ica-num 4 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_4
python main.py --attack ica --ica-num 5 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_5
python main.py --attack ica --ica-num 6 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_6
python main.py --attack ica --ica-num 7 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_7
python main.py --attack ica --ica-num 8 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_8
python main.py --attack ica --ica-num 9 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_9
python main.py --attack ica --ica-num 10 --model-path ./../Llama-2-7b-chat-hf --save-name llama_ICA_10

python main.py --attack ica --ica-num 1 --model-path ./../falcon --save-name falcon_ICA_1
python main.py --attack ica --ica-num 2 --model-path ./../falcon --save-name falcon_ICA_2
python main.py --attack ica --ica-num 3 --model-path ./../falcon --save-name falcon_ICA_3
python main.py --attack ica --ica-num 4 --model-path ./../falcon --save-name falcon_ICA_4
python main.py --attack ica --ica-num 5 --model-path ./../falcon --save-name falcon_ICA_5
python main.py --attack ica --ica-num 6 --model-path ./../falcon --save-name falcon_ICA_6
python main.py --attack ica --ica-num 7 --model-path ./../falcon --save-name falcon_ICA_7
python main.py --attack ica --ica-num 8 --model-path ./../falcon --save-name falcon_ICA_8
python main.py --attack ica --ica-num 9 --model-path ./../falcon --save-name falcon_ICA_9
python main.py --attack ica --ica-num 10 --model-path ./../falcon --save-name falcon_ICA_10

# No Attack
python main.py --attack none --model-path ./../falcon --save-name falcon_NA
python main.py --attack none --model-path ./../Llama-2-7b-chat-hf --save-name llama_NA
python main.py --attack none --model-path ./../vicuna-7b-v1.5 --save-name vicuna_NA