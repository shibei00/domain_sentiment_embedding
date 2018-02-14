# python train.py --domain1=books --domain2=music --vocabulary_size=10000 --num_sampled=5 --num_skips=2 --skip_window=1 --embedding_dimension=200 --batch_size=32 --num_steps=200000 --display_steps=1000 --e_steps=10000 --learning_rate=1.0
# python train.py --domain1=books --domain2=electronics --vocabulary_size=10000 --num_sampled=5 --num_skips=2 --skip_window=1 --embedding_dimension=200 --batch_size=32 --num_steps=200000 --display_steps=1000 --e_steps=10000 --learning_rate=1.0
# python train.py --domain1=electronics --domain2=music --vocabulary_size=10000 --num_sampled=5 --num_skips=2 --skip_window=1 --embedding_dimension=200 --batch_size=32 --num_steps=200000 --display_steps=1000 --e_steps=10000 --learning_rate=1.0
python train.py --domain1=books --domain2='kitchen_&_housewares' --vocabulary_size=10000 --num_sampled=5 --num_skips=2 --skip_window=1 --embedding_dimension=200 --batch_size=32 --num_steps=200000 --display_steps=1000 --e_steps=10000 --learning_rate=1.0
python train.py --domain1=music --domain2='kitchen_&_housewares' --vocabulary_size=10000 --num_sampled=5 --num_skips=2 --skip_window=1 --embedding_dimension=200 --batch_size=32 --num_steps=200000 --display_steps=1000 --e_steps=10000 --learning_rate=1.0
python train.py --domain1=electronics --domain2='kitchen_&_housewares' --vocabulary_size=10000 --num_sampled=5 --num_skips=2 --skip_window=1 --embedding_dimension=200 --batch_size=32 --num_steps=200000 --display_steps=1000 --e_steps=10000 --learning_rate=1.0


