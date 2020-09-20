import numpy as np

dim_in = 15
              # 入力は1次元
dim_out = 15           # 出力は1次元
hidden_count = 1024     # 隠れ層のノードは1024個
learn_rate = 0.001      # 学習率
id=0
cnt=0
line=0
word_list=[0]*dim_in
dictionary_list=[]
# 訓練データは x は -1～1、y は 2 * x ** 2 - 1
with open("dictionary.txt","r",encoding="utf-8") as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
dictionary_list = [x.strip() for x in content] 
print(dictionary_list)
train_count = 640        # 訓練データ数
fi=open("input2.txt","r",encoding="utf-8")
for row in fi:
    line=line+1
x=[[0]*dim_in for i in range(line)]
fi.close()
fi=open("input2.txt","r",encoding="utf-8")
length=len(dictionary_list)
cnt=0
for rowi in fi:  #行読み込み
    for j in range(dim_in): 
        x[cnt][j]=0 #入力初期化
        i=1
    for i in range(min(15,len(rowi.split()))): #単語読み込み
            for id in range(length):# 辞書参照
                if dictionary_list[id]==rowi.split()[i]:
                    word_list[j]=id
                    x[cnt][i]=id
                    break
                #print(id)
    print(cnt,line)
    cnt=cnt+1
print(x)
fi.close()
cnt=0
y=[[0]*dim_in for i in range(line)]
fo=open("output2.txt","r",encoding="utf-8")
length=len(dictionary_list)
for rowo in fo:  #行読み込み
    for j in range(dim_out): 
        y[cnt][j]=0#入力初期化
        i=1
    for i in range(min(dim_out,len(rowo.split()))): #単語読み込み
        for id in range(length):# 辞書参照
            if dictionary_list[id]==rowo.split()[i]:
                y[cnt][i]=id
                break
                #print(id)
    print(cnt,line)
    cnt=cnt+1
print(y)
fo.close()


#train_x = np.arange(-1, 1, 2 / train_count / dim_in).reshape((train_count, dim_in))
#train_y = np.array([2 * x ** 2 - 1 for x in train_x]).reshape((train_count, dim_out))
#print(train_x)
# 重みパラメータ。-0.5 〜 0.5 でランダムに初期化。この行列の値を学習する。
w1 = np.random.rand(hidden_count, dim_in) - 0.5
w2 = np.random.rand(dim_out, hidden_count) - 0.5
b1 = np.random.rand(hidden_count) - 0.5
b2 = np.random.rand(dim_out) - 0.5

# 活性化関数は ReLU
def activation(x):
    return np.maximum(0, x)

# 活性化関数の微分
def activation_dash(x):
    return (np.sign(x) + 1) / 2

# 順方向。学習結果の利用。
def forward(x):
    return w2 @ activation(w1 @ x + b1) + b2

# 逆方向。学習
def backward(x, diff):
    global w1, w2, b1, b2
    v1 = (diff @ w2) * activation_dash(w1 @ x + b1)
    v2 = activation(w1 @ x + b1)

    w1 -= learn_rate * np.outer(v1, x)  # outerは直積
    b1 -= learn_rate * v1
    w2 -= learn_rate * np.outer(diff, v2)
    b2 -= learn_rate * diff

# メイン処理
idxes = np.arange(train_count)          # idxes は 0～63
for epoc in range(2000):                # 1000エポック
    np.random.shuffle(idxes)            # 確率的勾配降下法のため、エポックごとにランダムにシャッフルする
    error = 0                           # 二乗和誤差
    for idx in idxes:
        y2 = forward(x[idx])       # 順方向で x から y を計算する
        diff = y2 - y[idx]         # 訓練データとの誤差
        error += np.linalg.det(diff) ** 2              # 二乗和誤差に蓄積
        backward(x[idx], diff)    # 誤差を学習
    print(error.sum())                  # エポックごとに二乗和誤差を出力。徐々に減衰して0に近づく。