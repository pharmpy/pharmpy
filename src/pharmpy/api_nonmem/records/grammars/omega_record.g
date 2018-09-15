// $OMEGA  [DIAGONAL(n)|BLOCK(n)|BLOCK(n) SAME(m)|BLOCK SAME(m)]
//         [BLOCK(n)VALUES(diag,odiag)]
//         [VARIANCE|STANDARD]  [COVARIANCE|CORRELATON] [CHOLESKY]
//         [[value1]  [value2]  [value3] ... [FIXED]]
//         [(value)xn]
//
// 4 legal forms
//   1. $OMEGA   [DIAGONAL(n)] [ v11 v22 v33 ... vnn ]
//   2. $OMEGA BLOCK(n) [ v11 v21 v22 v31 v32 v33 ... vn1 vn2 ... vnn ]
//   3. $OMEGA BLOCK(n) SAME(m)
//   4. $OMEGA BLOCK(n) VALUES(diag,odiag)

// Samples from nmhelp:
//   %input | BLOCK(3)  6. .005 .3 .0002 .006 .4|
//   %input | .04 .12|
//   %input | BLOCK(2) .04 .002 .12|
//   %input | BLOCK(2) SAME(3)|
//   %input | BLOCK(6) VALUES(0.1,0.01)|
//
// All these inputs have the same meaning:
//   %input | BLOCK(2) ; or $OMEGA VARIANCE COVARIANCE BLOCK(2)\n0.64\n-0.24 0.58|
//   %input | STANDARD BLOCK(2)\n0.8\n-0.24 0.762|
//   %input | STANDARD CORRELATION BLOCK(2)\n0.8\n-0.394 0.762|
//   %input | VARIANCE CORRELATION BLOCK(2)\n0.64\n-0.394 0.58|
//   %input | CHOLESKY BLOCK(2)\n0.8\n-0.3 0.7|

root : [_opt] [diagonal] [_opt] (_item)+ [ws] // form 1
     | [_opt] block [_opt] (_item)+ [ws]      // form 2
     | [_opt] block [_opt] same (_opt | [ws] comment)* [ws]    // form 3
     | [_opt] block [_opt] values (_opt | [ws] comment)* [ws]  // form 4

// repeat of init or comment (form 1 & 2)
_item : [ws] omega | [ws] comment

// can be tied to WHOLE record or ONE INIT
_opt     : (ws (FIX | VAR | SD | CORR | COV | CHOLESKY))* [ws]  // note ws injection as 1st child
FIX      : "FIXED" | "FIX"
VAR      : "VARIANCE"
SD       : "STANDARD" | "SD"
CORR     : "CORRELATION"
COV      : "COVARIANCE"
CHOLESKY : "CHOLESKY"

// parentheses sizes (optional for SAME and BLOCK with SAME)
block    : "BLOCK" _lpar size _rpar
same     : "SAME" _lpar size _rpar
diagonal : ("DIAGONAL" | "DIAG") _lpar size _rpar
values   : "VALUES" _lpar diag sep odiag _rpar

size  : INT
diag  : NUMERIC
odiag : NUMERIC

// separator and clustering of init values
sep : WS | [WS] "," [WS]
_lpar : "(" [WS]
_rpar : [WS] ")"

// omega init value
omega : init _opt*
      | _lpar [WS] init _opt* _rpar [n]
      | _lpar _opt* init [WS] _rpar [n]

init  : NUMERIC
n     : "x" INT

// common misc rules
ws      : WS_ALL
comment : ";" [WS] [COMMENT]

// common misc terminals
WS: (" " | /\t/)+
WS_ALL: /\s+/

// common naked/enquoted text terminals
COMMENT : /\S.*(?<!\s)/  // no left/right whitespace padding

// common numeric terminals
DIGIT: "0".."9"
INT: DIGIT+
DECIMAL: (INT "." [INT] | "." INT)
SIGNED_INT: ["+" | "-"] INT

EXP: "E" SIGNED_INT
FLOAT: (INT EXP | DECIMAL [EXP])
SIGNED_FLOAT: ["+" | "-"] FLOAT

NUMBER: (FLOAT | INT)
SIGNED_NUMBER: ["+" | "-"] NUMBER
NUMERIC: (NUMBER | SIGNED_NUMBER)
