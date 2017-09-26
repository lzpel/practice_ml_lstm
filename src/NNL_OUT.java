/**
 * Created by user on 2017/08/23.
 * 出力層を表す
 */
public class NNL_OUT extends NNL {
	static final int ACTIVE_TANH=0;
	static final int ACTIVE_RELU=1;
	static final int ACTIVE_SMAX=2;
	static final int ACTIVE_SIGM=3;
	static final int ACTIVE_IDEN=4;
	NNN W;
	NNV vi,vo[];
	int t;
	NNL_OUT(NNL m){
		super(m.si,m.so);
		initvector();
	}
	NNL_OUT(int size_i,int size_o,int active_type){
		super(size_i,size_o);
		initvector();
		t=active_type;
		W=new NNN(si,so);
	}
	public void initvector(){
		vi=new NNV(si);
		vo=NNV.newmulti(2,so);
	}
	public void propagate(boolean f){
		if(f){
			NNV.vcross(f,vo[0],vi,W);
			if(t==ACTIVE_SMAX)NNV.vsmaxactive(f,vo[1],vo[0]);
			if(t==ACTIVE_SIGM)NNV.vsigmactive(f,vo[1],vo[0]);
			if(t==ACTIVE_IDEN)NNV.videnactive(f,vo[1],vo[0]);
		}else{
			//getlossでvo[0]のδが求まっているので活性化関数の誤差逆伝播は書かない
			NNV.vcross(f,vo[0],vi,W);
		}
	}
	public void learn(float r){
		NNV.nlearn(vo[0],vi,W,r);
	}
	public float getloss(NNV d){
		if(t==ACTIVE_SMAX)return NNV.vsmaxloss(d,vo[1],vo[0]);
		if(t==ACTIVE_SIGM)return NNV.vsigmloss(d,vo[1],vo[0]);
		if(t==ACTIVE_IDEN)return NNV.videnloss(d,vo[1],vo[0]);
		System.out.println("E:You can use softmax, sigmoid and identity function as activation function to get loss");
		return 0;
	}
	public void lcpy(boolean f,NNL i0){
		NNL_OUT i=(NNL_OUT)i0;
		NNV.vcpy(f,vi,i.vi);
		for(int n=0;n<vo.length;n++)NNV.vcpy(f,vo[n],i.vo[n]);
	}
	public NNV geti() {
		return vi;
	}
	public NNV geto(){
		return vo[1];
	}
	public void zerodelta(){
		vi.zerodelta();
		for(int n=0;n<vo.length;n++)vo[n].zerodelta();
	}
	public void zerovalue(){
		vi.zerovalue();
		for(int n=0;n<vo.length;n++)vo[n].zerovalue();
	}
}
