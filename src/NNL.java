/**
 * Created by user on 2017/08/22.
 * 層を表す
 */
public class NNL {
	int si,so;
	NNL(int size_i,int size_o){
		si=size_i;
		so=size_o;
	}
	public void initvector(){
		//行列ベクトル確保
	}
	public void propagate(boolean f){
		//伝播計算
	}
	public void learn(float rate){
		//修正
	}
	public void lcpy(boolean f,NNL i0){
	}
	public float getloss(NNV t){
		return 0;
	}
	public NNV geti(){
		return null;
	}
	public NNV geto(){
		return null;
	}
	public void zerodelta(){}
	public void zerovalue(){}
}

/*
t=[]
while True:
	tmp=raw_input()
	if tmp:
		t.append(tmp)
	else:
		print("\n".join(t[::-1]))
		break
 */