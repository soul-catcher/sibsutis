#include <iostream>

using namespace std;

const int n = 10;
int Aw[n+1][n+1] = { 0 }; //матрица весов
int Ap[n+1][n+1] = { 0 };
int Ar[n+1][n+1] = { 0 };
int w[n+1];
int V[n+1];

struct Vertex
{
    int data;
    int W;
    Vertex *left;
    Vertex *right;
} *rootS, *rootS1, *rootS2;

//void GoToXY(short x, short y)
//{
//    HANDLE StdOut = GetStdHandle(STD_OUTPUT_HANDLE);
//    COORD coord = { x, y };
//    SetConsoleCursorPosition(StdOut, coord);
//}
//void print(Vertex* p, int x, int y, int s, int h)
//{
//    if (p)
//    {
//        if (s == 0)
//        {
//            GoToXY(x, y);
//        }
//        else if (s == -1)
//        {
//            GoToXY(x + 1, y - 2); cout << "/";
//            GoToXY(x, y);
//        }
//        else if (s == 1)
//        {
//            GoToXY(x - 1, y - 2); printf("%c", 92);
//            GoToXY(x, y);
//        }
//        cout << p->data;
//        print(p->left, x - pow(2, h), y += 3, -1, h - 1);
//        print(p->right, x + pow(2, h), y, 1, h - 1);
//    }
//}


int htree(Vertex* p, int &count)
{
    if (p == nullptr) count = 0;
    else count = 1 + max(htree(p->left, count), htree(p->right, count));
    return count;
}
float medsizetree(Vertex* p, int L, float &count)
{
    if (p == nullptr) count = 0;
    else count = p->W * L + medsizetree(p->left, L + 1, count) + medsizetree(p->right, L + 1, count);
    return count;
}
int sum(Vertex *p, int &count)
{
    if (p == nullptr) count = 0;
    else count = p->data + sum(p->left, count) + sum(p->right, count);
    return count;
}
int sizeoftree(Vertex* p, int &count)
{
    if (p == nullptr) count = 0;
    else count = 1 + sizeoftree(p->left, count) + sizeoftree(p->right, count);
    return count;
}
int Weighttree(Vertex* p, int &count)
{
    if (p == nullptr) count = 0;
    else count = p->W + Weighttree(p->left, count) + Weighttree(p->right, count);
    return count;
}


void SDPREC(int D, int wei, Vertex *&p)
{
    if (!p)
    {
        p = new Vertex;
        p->data = D;
        p->W = wei;
        p->left = nullptr;
        p->right = nullptr;
    }
    else if (D < p->data) SDPREC(D, wei, p->left);
    else if (D > p->data) SDPREC(D, wei, p->right);
}
void AW()
{
    for (int i = 0; i <= n; i++)
        for (int j = i + 1; j <= n; j++)
            Aw[i][j] = Aw[i][j - 1] + w[j];


}

void AP_AR()
{
    int i, j, m, min, k, h;
    for (i = 0; i <= n - 1; i++)
    {
        j = i + 1;
        Ap[i][j] = Aw[i][j];
        Ar[i][j] = j;
    }
    for (h = 2; h <= n; h++)
    {
        for (i = 0; i <= n - h; i++)
        {
            j = i + h;
            m = Ar[i][j - 1];
            min = Ap[i][m - 1] + Ap[m][j];
            for (k = m + 1; k <= Ar[i + 1][j]; k++)
            {
                int x = Ap[i][k - 1] + Ap[k][j];
                if (x < min)
                {
                    m = k;
                    min = x;
                }
            }
            Ap[i][j] = min + Aw[i][j];
            Ar[i][j] = m;
        }
    }
}
void matr(int matrix[n+1][n+1])
{
    for (int i = 0; i <= n; i++)
    {
        for (int j = 0; j <= n; j++)
            cout << matrix[i][j] << "  ";
        cout << endl;
    }
}
void Create_Tree(int l, int r)
{
    if (l < r) {
        int k = Ar[l][r];
        SDPREC(V[k], w[k], rootS);
        Create_Tree(l, k - 1);
        Create_Tree(k, r);
    }
}
void A1()
{
    for (int i =1; i <= n; i++)
        SDPREC(V[i], w[i], rootS1);
}
void A2(int L, int R)
{
    int wes = 0, sum = 0;
    int i;
    if (L <= R)
    {
        for (i = L; i <= R; i++)
            wes += w[i];
        for (i = L; i <= R - 1; i++)
        {
            if ((sum <( wes / 2)) && ((sum + w[i])>(wes / 2))) break;
            sum += w[i];
        }
        SDPREC(V[i], w[i], rootS2);
        A2(L, i - 1);
        A2(i + 1, R);
    }
}
void Print(Vertex *p)
{
    if (p)
    {
        Print(p->left);
        printf("%d ", p->data);
        Print(p->right);
    }
}

void FillInc()
{
    w[0] = 0;
    V[0] = 0;
    for (int i = 1; i <= n; i++)
    {
        V[i] = i;

        w[i] = rand() % 100;
    }
}

void PrintMas() {
    for (int i = 0; i <= n; i++)
    {
        cout << V[i] << " ";
    }
    cout << endl << endl;
    for (int i = 0; i <= n; i++)
    {
        cout << w[i] << " ";
    }
}

int main()
{
    setlocale(LC_ALL, "rus");
    srand(0);
    FillInc();
    PrintMas();
    cout << endl << endl;

    rootS = new Vertex;
    rootS->data = V[1];
    rootS->W = w[1];
    rootS->left = nullptr;
    rootS->right = nullptr;
    AW();
    matr(Aw);
    AP_AR();
    cout << endl;
    matr(Ap);
    cout << endl;
    matr(Ar);
    cout << endl;
    Create_Tree(1, n);
    Print(rootS);
    cout << endl;

    rootS1 = new Vertex;
    rootS1->data = V[1];
    rootS1->W = w[1];
    rootS1->left = nullptr;
    rootS1->right = nullptr;
    A1();
    Print(rootS);
    cout << endl;

    rootS2 = new Vertex;
    rootS2->data = V[1];
    rootS2->W = w[1];
    rootS2->left = nullptr;
    rootS2->right = nullptr;
    A2(1, n);
    Print(rootS);
    cout << endl;


    cout << endl << endl << endl << endl;
    Vertex* tmp = rootS;
    float count1 = 0;
    int count = 0;
    //print(tmp, 50, 25, 0, htree(rootS, count) - 1);
    //cout << endl << endl << endl << endl;
    //cout << endl << endl << endl << endl;
    //cout << endl << endl << endl << endl;
    int W = 0;
    cout << Weighttree(rootS, W) << endl;
    cout << medsizetree(rootS, 1, count1) << endl;

    printf("n=100  | Размер | Контр.сумма | Средн.высота |\n");
    printf("-------|--------|-------------|--------------|\n");
    printf("  ДОП  |%8d|%13d|%14.2f|\n", sizeoftree(rootS, count), sum(rootS, count), ((medsizetree(rootS, 1, count1)) / Weighttree(rootS, W)));
    printf("  A 1  |%8d|%13d|%14.2f|\n", sizeoftree(rootS1, count), sum(rootS1, count), (medsizetree(rootS1, 1, count1) / Weighttree(rootS, W)));
    printf("  A 2  |%8d|%13d|%14.2f|\n", sizeoftree(rootS2, count), sum(rootS2, count), (medsizetree(rootS2, 1, count1) / Weighttree(rootS, W)));


//    system("pause");
//    return 0;
}
