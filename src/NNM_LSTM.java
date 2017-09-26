/**
 * Created by user on 2017/08/25.
 */
public class NNM_LSTM extends NNM{
	NNL l0,l1;
	NNL[] l0l,l1l;
	int l;
	float[] vi,vo,vd;
	NNV vt;
	NNM_LSTM(int size_i,int size_m,int size_o,int active,int length){
		super(size_i,size_m,size_o);
		l=length;
		vi=new float[l*si];
		vo=new float[l*so];
		vd=new float[l*so];
		vt=new NNV(so);
		l0=new NNL_LSTM(si,sm);
		l0l=new NNL[l];
		for(int i=0;i<l;i++){
			l0l[i]=new NNL_LSTM(l0);
		}
		l1=new NNL_OUT(sm,so,active);
		l1l=new NNL[l];
		for(int i=0;i<l;i++){
			l1l[i]=new NNL_OUT(l1);
		}
	}
	public float learn(float rate){
		if(rate>=0){
			float loss=0;
			l1.zerovalue();
			l0.zerovalue();
			for(int n=0;n<l;n++){
				for(int m=0;m<si;m++){
					l0.geti().v[m]=vi[si*n+m];
				}
				l0.propagate(true);
				NNV.vcpy(true, l1.geti(), l0.geto());
				l1.propagate(true);
				for(int m=0;m<so;m++){
					vo[so*n+m]=l1.geto().v[m];
				}
				l1l[n].lcpy(true,l1);
				l0l[n].lcpy(true,l0);
			}
			l1.zerodelta();
			l0.zerodelta();
			for (int n=l-1; n >= 0; n--) {
				l1.lcpy(true,l1l[n]);
				l0.lcpy(true,l0l[n]);
				//ここから
				for(int m=0;m<so;m++){
					vt.v[m]=vd[so*n+m];
				}
				loss=l1.getloss(vt);
				//ここまで
				l1.propagate(false);
				NNV.vcpy(false, l1.geti(), l0.geto());
				l0.propagate(false);
				l1.lcpy(false,l1l[n]);
				l0.lcpy(false,l0l[n]);
			}
			for(int n=0;n<l;n++) {
				l1.lcpy(true,l1l[n]);
				l0.lcpy(true,l0l[n]);
				l1l[n].lcpy(false,l1);
				l0l[n].lcpy(false,l0);
				l1.learn(rate);
				l0.learn(rate);
			}
			return loss;
		}else{
			for(int i=0;i<si;i++){
				l0.geti().v[i]=vi[i];
			}
			l0.propagate(true);
			NNV.vcpy(true, l1.geti(), l0.geto());
			l1.propagate(true);
			for(int i=0;i<so;i++){
				vo[i]=l1.geto().v[i];
			}
			return 0;
		}
	}
	public float[] geti(){
		return vi;
	}
	public float[] geto(){
		return vo;
	}
	public float[] getd(){
		return vd;
	}
	public static void test(){
		int T=10;
		NNM_LSTM nnm=new NNM_LSTM(1,10,1, NNL_OUT.ACTIVE_IDEN,50);
		for(int m=0;m<2000;m++){
			for(int n=0;n<nnm.l;n++){
				nnm.geti()[n]= (float) Math.sin(2*Math.PI*(n+m+0)/T);
				nnm.getd()[n]= (float) Math.sin(2*Math.PI*(n+m+1)/T);
			}
			nnm.learn(0.01f);
		}
		for(int m=0;m<100;m++){
			if(m<50){
				nnm.geti()[0]= (float) Math.sin(2*Math.PI*(m)/T);
			}else{
				nnm.geti()[0]=nnm.geto()[0];
			}
			nnm.learn();
			System.out.println(nnm.geto()[0]);
		}
	}
}
