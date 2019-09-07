#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;
int s;
double C[50][10],L[50];

float chas[50]={0};
char ch[50];

void Hen(int n){
    int i,j;
    double  Q[n];
    Q[0]=0.0;
    L[0]=(int)(-log2(chas[0])) + 1 ;
    for(i=1;i<n;i++){
        Q[i]= Q[i-1] + chas[i-1];
        L[i] = (int)(-log2(chas[i])) + 1 ;
    }
    cout << endl;
    for (i=0; i<n; i++){
        for (j=0;j<L[i];j++){
            Q[i] = Q[i]*2;
            C[i][j] = int(Q[i]);
            if (Q[i] >1) Q[i] = Q[i] - 1 ;
        }
    }
}

void BubbleSort2 (float *A, int n){
    int i,j;
    float t;
    char t1;
    for (i=0; i<n-1; i++){
        for (j=0; j<n-1; j++)
            if (A[j+1]>A[j]){
                t=A[j+1];
                A[j+1]=A[j];
                A[j]=t;
                t1=ch[j+1];
                ch[j+1]=ch[j];
                ch[j]=t1;
            }
    }
}

int main(){
    int j=0, q=0, i, kol = 0, sum = 0;
    double H = 0, Lsr=0.0;
    ifstream f("1.txt", ios::out | ios::binary);
    map<char,int> m;
    while (!f.eof()){
        char c = f.get();
        if ((c==10)||(c==13)) continue;
        m[c]++;
        kol++;
    }
    s = m.size();
    for(map<char,int>::iterator itr=m.begin(); itr!=m.end(); j++, ++itr){
        ch[j] = itr->first;
        chas[j] = (float)itr->second/kol;
    }
    BubbleSort2 (chas,51);
    Hen(50);
    cout<<"Символ "<<"Верояность символа  "<<"Код символа"<<endl;
    for (i=0;i<50;i++){
        cout<<"    "<<ch[i]<<"     ";
        printf("%.8f",chas[i]);
        cout<<"       ";
        for (j=0;j<L[i];j++) cout<<C[i][j];
        cout<<endl;
    }
    for (i=0;i<s;i++) H=H-chas[i]*(log(chas[i])/log(2.0));
    for (i=0;i<s;i++) Lsr=Lsr+chas[i]*L[i];
    cout << "Kol-vo = " << kol << endl;
    cout << "Size = " << s << endl;
    cout<<"Энтропия исходного файла ровна "<<H<<endl;
    cout<<"Средняя длина кодового слова ровна "<<Lsr<<endl;
    cout<<"Соотношение "<<Lsr<<"<"<<H<<"+ 1"<<" выполнено"<<endl;
    return 0;
}
