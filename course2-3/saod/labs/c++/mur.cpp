#include <iostream>
#include <vector>
#include <map>
#include <list>
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;
int s;
double C[51][13],L[51];

void GB(int n, double ver[]){
    int i,j;
    double  Q[n],pr=0;
    for(i=0;i<n;i++){
        Q[i]= pr + ver[i]/2;
        pr = pr + ver[i];
        L[i] = (int)(-log2(ver[i])) + 2 ;
    }
    for (i=0; i<n; i++){
        for (j=0;j<L[i];j++){
            Q[i] = Q[i]*2;
            C[i][j] = int(Q[i]);
            if (Q[i] >1) Q[i] = Q[i] - 1 ;
        }
    }
}

int main(){
    int j=0, q=0, kol = 0, ch[51], sum = 0 ;
    char simbol[51];
    double ver[75], codee[75],longer=0, H = 0;
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
        simbol[j] = itr->first;
        ch[j] = itr->second;
        ver[j] = (double)itr->second/kol;
    }
    GB(s, ver);
    for (int i=0;i<49;i++){
        cout << simbol[i] << " = " ;
        cout << setw(1);
        for (j=0; j<L[i]; j++) cout <<  C[i][j];
        cout << endl;
    }
    for (j=0; j<m.size(); j++){
        H += ver[j] * log2l(ver[j]);
        longer +=ver[j] * L[j];
    }
    cout << "Kol-vo = " << kol << endl;
    cout << "Size = " << s << endl;
    cout << "Entropy = " << H*(-1) << endl;
    cout << "Average word length = " << longer << endl;
    return 0;
}
