(defun f (ld lst1 lst2)
    (if (and lst1 lst2)
    (cons
        (funcall ld (car lst1) (car lst2))
        (f ld (cdr lst1) (cdr lst2))
    )
    )
)

(princ(f (lambda (x y) (/ x y)) '(10 14 8) '(5 14 4)))

(print(f (lambda (x y) (if (> x y) x y)) '(10 14 8) '(5 14 4)))
