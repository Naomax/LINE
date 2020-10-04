import numpy as np 
import MeCab
 
denom=1000000 #denominator
dim_in = 15
              # 入力は1次元
dim_out = 15           # 出力は1次元
hidden_count = 1024     # 隠れ層のノードは1024個
learn_rate = 0.001      # 学習率
id=0
cnt=0
line=0
word_list=[0]*dim_in
x_test_parse_id=[0]*dim_in
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
    for i in range(min(15,len(rowi.split()))): #単語読み込み
            for id in range(length):# 辞書参照
                if dictionary_list[id]==rowi.split()[i]:
                    x[cnt][i]=id/denom
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
    for i in range(min(dim_out,len(rowo.split()))): #単語読み込み
        for id in range(length):# 辞書参照
            if dictionary_list[id]==rowo.split()[i]:
                y[cnt][i]=id/denom
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
def square_sum(arg):
    sum=0
    for i in range(len(arg)):
        sum=sum+np.square(arg[i])
    #print(sum,arg)
    return sum
# メイン処理
idxes = np.arange(train_count)          # idxes は 0～63
for epoc in range(2000):                # 1000エポック
    np.random.shuffle(idxes)            # 確率的勾配降下法のため、エポックごとにランダムにシャッフルする
    error = 0                           # 二乗和誤差
    for idx in idxes:
        y2 = forward(x[idx])       # 順方向で x から y を計算する
        diff = y2 - y[idx]         # 訓練データとの誤差
        error += square_sum(diff)              # 二乗和誤差に蓄積
        backward(x[idx], diff)    # 誤差を学習
    print(error.sum())		# エポックごとに二乗和誤差を出力。徐々に減衰して0に近づく。
    while True:
        x_test=input()
        if x_test=="おわり":
            break
        #構文解析
        tagger=MeCab.Tagger("-Owakati")
        x_test_parse=tagger.parse(x_test)
        with open("dictionary.txt","r",encoding="utf-8") as f:
            content=f.readlines()
            dictionary_list=[x.strip() for x in content]
        #x_test_parse_id初期化
        for j in range(dim_in):
            x_test_parse_id[j]=0
        s2_list=[]
        s2=""
        for i in range(len(x_test_parse)):
            if (x_test_parse[i]==" ") or (x_test_parse[i]=="　"):
                s2_list.append(s2)
                s2=""
                continue
            s2=s2+x_test_parse[i]   
        for i in range(len(s2_list)):
            flag=0
            for id in range(len(dictionary_list)):
                if(s2_list[i]==dictionary_list[id]):
                    x_test_parse_id[i]=id/denom
                    flag=1
                    break
            if flag==0:
                with open("dictionary.txt","a",encoding="utf-8") as f:
                    print(s2_list[i],dictionary_list[id])
                    f.write(s2_list[i]+'\n')
                    dictionary_list.append(s2_list[i])
                    x_test_parse_id[i]=len(dictionary_list[i])/denom
        #dictionary_listのidを百万で割る
        y3=forward(x_test_parse_id)
        dictionary_list2=[]
        for i in range(len(dictionary_list)):
            dictionary_list2.append(i/denom)
        #近い数字探索
        for i in range(dim_out):
            min=abs(y3[i]-dictionary_list2[0])
            min_id=0
            for j in range(len(dictionary_list2)):
                min=abs(y3[i]-dictionary_list2[j])
                min_id=j
            y4=[]
            y4.append(dictionary_list[min_id])
        print(y4)

