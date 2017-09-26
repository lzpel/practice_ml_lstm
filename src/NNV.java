/**
 * Created by user on 2017/08/22.
 * ベクトルを表すクラス
 */
public class NNV{
	float v[],d[];
	int s;
	NNV(int size){
		s=size;
		v=new float[s];
		d=new float[s];
	}
	public void zerovalue(){
		for (int i = 0; i < s; i++) {
			v[i]=0;
		}
	}
	public void zerodelta(){
		for (int i = 0; i < s; i++) {
			d[i]=0;
		}
	}
	public static void print(NNV i0,NNV i1){
		for(int i = 0; i< i0.s; i++){
			if(i0!=null) {
				System.out.print(i0.v[i]);
			}
			if(i1!=null){
				System.out.print(",");
				System.out.print(i1.v[i]);
			}
			System.out.println();
		}
	}
	public static NNV[] newmulti(int count,int size){
		NNV[] a=new NNV[count];
		for(int i=0;i<count;i++){
			a[i]=new NNV(size);
		}
		return a;
	}
	public static void nlearn(NNV o0, NNV i0, NNN n, float rate){
		for (int i = 0; i <= n.si; i++) {
			for (int o = 0; o < n.so; o++) {
				n.v[o * (n.si+1) + i] += ((i==n.si)?1:i0.v[i]) * o0.d[o] * (-rate);
			}
		}
	}
	public static void vcross(boolean f,NNV o0,NNV i0,NNN n0){
		if(f) {
			for (int o = 0; o < n0.so; o++) {
				float tmp = n0.v[o * (n0.si+1) + n0.si];
				for (int i = 0; i < n0.si; i++) {
					tmp += i0.v[i] * n0.v[o * (n0.si+1) + i];
				}
				o0.v[o] = tmp;
			}
		}else {
			for (int i = 0; i < n0.si; i++) {
				float tmp = 0;
				for (int o = 0; o < n0.so; o++) {
					tmp += o0.d[o] * n0.v[o * (n0.si+1) + i];
				}
				i0.d[i] = tmp;
			}
		}
	}
	public static void vcpy(boolean f,NNV o0,NNV i0){
		if(f){
			for(int i = 0; i<i0.s; i++){
				o0.v[i]=i0.v[i];
			}
		}else{
			for(int i = 0; i<i0.s; i++){
				i0.d[i]=o0.d[i];
			}
		}
	}
	public static void vcpy(boolean f,NNV o0,NNV o1,NNV o2,NNV o3,NNV o4,NNV i0){
		if(f){
			for(int i = 0; i<i0.s; i++){
				if(o0!=null)o0.v[i]=i0.v[i];
				if(o1!=null)o1.v[i]=i0.v[i];
				if(o2!=null)o2.v[i]=i0.v[i];
				if(o3!=null)o3.v[i]=i0.v[i];
				if(o4!=null)o4.v[i]=i0.v[i];
			}
		}else{
			for(int i = 0; i<i0.s; i++){
				float tmp=0;
				if(o0!=null)tmp+=o0.d[i];
				if(o1!=null)tmp+=o1.d[i];
				if(o2!=null)tmp+=o2.d[i];
				if(o3!=null)tmp+=o3.d[i];
				if(o4!=null)tmp+=o4.d[i];
				i0.d[i]=tmp;
			}
		}
	}
	public static void vsum(boolean f,NNV o0,NNV i0,NNV i1){
		if(f){
			for(int i = 0; i<o0.s; i++){
				o0.v[i]=i0.v[i]+i1.v[i];
			}
		}else{
			for(int i = 0; i<o0.s; i++){
				i0.d[i]=i1.d[i]=o0.d[i];
			}
		}
	}
	public static void vmul(boolean f,NNV o0,NNV i0,NNV i1){
		if(f){
			for(int i = 0; i<o0.s; i++){
				o0.v[i]=i0.v[i]*i1.v[i];
			}
		}else{
			for(int i = 0; i<o0.s; i++){
				i0.d[i]=o0.d[i]*i1.v[i];
				i1.d[i]=o0.d[i]*i0.v[i];
			}
		}
	}
	public static void vsigmactive(boolean f,NNV o0,NNV i0){
		if(f){
			for(int i = 0; i<o0.s; i++){
				o0.v[i]= (float) (1/(1+Math.exp(-i0.v[i])));
			}
		}else{
			for(int i = 0; i<o0.s; i++){
				i0.d[i]=o0.v[i]*(1-o0.v[i])*o0.d[i];
			}
		}
	}
	public static void vsmaxactive(boolean f,NNV o0,NNV i0){
		if(f){
			float max=-Float.MAX_VALUE;
			for(int i = 0; i<i0.s; i++) {
				if (i0.v[i] > max) {
					max = i0.v[i];
				}
			}
			float sum=0;
			for(int i = 0; i<i0.s; i++) {
				sum+=o0.v[i]=(float)Math.exp(i0.v[i]-max);
			}
			for(int i = 0; i<o0.s; i++) {
				o0.v[i]/=sum;
			}
		}
	}
	public static void videnactive(boolean f,NNV o0,NNV i0){
		if(f){
			for(int i = 0; i<o0.s; i++){
				o0.v[i]=i0.v[i];
			}
		}else{
			for(int i = 0; i<o0.s; i++){
				i0.d[i]=o0.d[i];
			}
		}
	}
	public static void vreluactive(boolean f,NNV o0,NNV i0){
		if(f){
			for(int i = 0; i<o0.s; i++){
				o0.v[i]=(i0.v[i]>0)?i0.v[i]:0;
			}
		}else{
			for(int i = 0; i<o0.s; i++){
				i0.d[i]=(i0.v[i]>0)?o0.d[i]:0;
			}
		}
	}
	public static void vtanhactive(boolean f,NNV o0,NNV i0){
		if(f){
			for(int i = 0; i<o0.s; i++){
				o0.v[i]=(float)Math.tanh(i0.v[i]);
			}
		}else{
			for(int i = 0; i<o0.s; i++){
				i0.d[i]=(1-o0.v[i]*o0.v[i])*o0.d[i];
			}
		}
	}
	public static float vsigmloss(NNV t0,NNV o0,NNV i0){
		float tmp=0;
		for(int i = 0; i<t0.s; i++) {
			i0.d[i] = o0.v[i] - t0.v[i];
			tmp+= - t0.v[i]*Math.log(o0.v[i])-(1-t0.v[i])*Math.log(1-o0.v[i]);
		}
		return tmp;
	}
	public static float vsmaxloss(NNV t0,NNV o0,NNV i0){
		float tmp=0;
		for(int i = 0; i<t0.s; i++) {
			i0.d[i] = o0.v[i] - t0.v[i];
			tmp+= - t0.v[i]*Math.log(o0.v[i]);
		}
		return tmp;
	}
	public static float videnloss(NNV t0,NNV o0,NNV i0){
		float tmp=0;
		for(int i = 0; i<t0.s; i++) {
			i0.d[i] = o0.v[i] - t0.v[i];
			tmp+= (o0.v[i] - t0.v[i])*(o0.v[i] - t0.v[i])/2;
		}
		return tmp;
	}
}