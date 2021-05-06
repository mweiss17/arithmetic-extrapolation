

python -u run.py --seed=1 --exp_name a --lr 0.01 --optimizer_type adam &
python -u run.py --seed=2 --exp_name a --lr 0.01 --optimizer_type adam &
python -u run.py --seed=3 --exp_name a --lr 0.01 --optimizer_type adam &

python -u run.py --seed=1 --exp_name a --lr 0.025 --optimizer_type adam &
python -u run.py --seed=2 --exp_name a --lr 0.025 --optimizer_type adam &
python -u run.py --seed=3 --exp_name a --lr 0.025 --optimizer_type adam &

python -u run.py --seed=1 --exp_name a --lr 0.05 --optimizer_type adam &
python -u run.py --seed=2 --exp_name a --lr 0.05 --optimizer_type adam &
python -u run.py --seed=3 --exp_name a --lr 0.05 --optimizer_type adam &

python -u run.py --seed=1 --exp_name a --lr 0.0075 --optimizer_type adam &
python -u run.py --seed=2 --exp_name a --lr 0.0075 --optimizer_type adam &
python -u run.py --seed=3 --exp_name a --lr 0.0075 --optimizer_type adam &

python -u run.py --seed=1 --exp_name a --lr 0.0005 --optimizer_type adam &
python -u run.py --seed=2 --exp_name a --lr 0.0005 --optimizer_type adam &
python -u run.py --seed=3 --exp_name a --lr 0.0005 --optimizer_type adam &
