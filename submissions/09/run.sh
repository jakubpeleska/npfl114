python3 homr_competition.py --weights_file=homr_lr0003_5LSTM1024d05_skip.h5 --skip_connections --lr=0.0003 --rnn=LSTM --rnn_dim=1024
python3 homr_competition.py --weights_file=homr_lr0003_5LSTM512d05_skip.h5 --skip_connections --lr=0.0003 --rnn=LSTM --rnn_dim=512
python3 homr_competition.py --weights_file=homr_lr0003_5GRU512d05_h128_skip.h5 --epochs=5 --skip_connections --lr=0.0003 --rnn=GRU --rnn_dim=512 --height=128
python3 homr_competition.py --weights_file=homr_lr0003_5GRU512d05_h128_cnn5_skip.h5 --epochs=5 --skip_connections --lr=0.0003 --rnn=GRU --rnn_dim=512 --height=128 --cnn_layers=5
python3 homr_competition.py --weights_file=homr_lr0003_3LSTM2048d01.h5 --lr=0.0003 --rnn=LSTM --rnn_dim=2048 --dropout=0.1 --rnn_layers=3
python3 homr_competition.py --weights_file=homr_lr0003_3LSTM512d01.h5 --lr=0.0003 --rnn=LSTM --rnn_dim=512 --dropout=0.1 --rnn_layers=3