# GraphDrone.fit Classification Portfolio Leaderboard

Public benchmark surface:

- `GraphDrone`
- `TabPFN`
- `TabM`

`TabR` is intentionally excluded from this public leaderboard. It remains a diagnostic-only runner in this branch.

| Dataset | Model | Mean Accuracy | Mean Macro F1 | Mean ROC-AUC | Mean PR-AUC | Mean Log Loss |
|---|---|---:|---:|---:|---:|---:|
| anneal | GraphDrone | 0.991093 | 0.911824 | 0.999145 | 0.971888 | 0.024683 |
| anneal | TabPFN | 0.991093 | 0.949586 | 0.999779 | 0.990728 | 0.024972 |
| anneal | TabM | 0.973270 | 0.866554 | 0.977438 | 0.944948 | 0.121049 |
| apsfailure | GraphDrone | 0.992947 | 0.886262 | 0.991590 | 0.878644 | 0.020579 |
| apsfailure | TabPFN | 0.994434 | 0.916444 | 0.992324 | 0.906569 | 0.017397 |
| apsfailure | TabM | 0.991816 | 0.875440 | 0.985869 | 0.828938 | 0.030976 |
| bank_customer_churn | GraphDrone | 0.866600 | 0.759156 | 0.870645 | 0.715697 | 0.328867 |
| bank_customer_churn | TabPFN | 0.867400 | 0.764103 | 0.872547 | 0.718045 | 0.328113 |
| bank_customer_churn | TabM | 0.857300 | 0.752675 | 0.858892 | 0.687440 | 0.344228 |
| bank_marketing | GraphDrone | 0.893079 | 0.610455 | 0.761346 | 0.392965 | 0.304330 |
| bank_marketing | TabPFN | 0.893256 | 0.610889 | 0.763381 | 0.393817 | 0.304226 |
| bank_marketing | TabM | 0.892017 | 0.619806 | 0.757223 | 0.385188 | 0.306009 |
| bioresponse | GraphDrone | 0.771529 | 0.770202 | 0.849326 | 0.861821 | 0.483568 |
| bioresponse | TabPFN | 0.788854 | 0.786775 | 0.866898 | 0.879451 | 0.458900 |
| bioresponse | TabM | 0.762729 | 0.760474 | 0.829553 | 0.830932 | 0.840866 |
| diabetes | GraphDrone | 0.769531 | 0.735065 | 0.845646 | 0.733209 | 0.457847 |
| diabetes | TabPFN | 0.770833 | 0.736253 | 0.847023 | 0.737953 | 0.456617 |
| diabetes | TabM | 0.746094 | 0.705953 | 0.825420 | 0.713278 | 0.544359 |
| maternal_health_risk | GraphDrone | 0.816568 | 0.818530 | 0.932357 | 0.882970 | 0.468217 |
| maternal_health_risk | TabPFN | 0.804734 | 0.808080 | 0.931446 | 0.883153 | 0.470264 |
| maternal_health_risk | TabM | 0.647929 | 0.638673 | 0.820203 | 0.713338 | 0.745089 |
| students_dropout_and_academic_success | GraphDrone | 0.783003 | 0.709348 | 0.894402 | 0.774014 | 0.547415 |
| students_dropout_and_academic_success | TabPFN | 0.787298 | 0.717279 | 0.898157 | 0.781421 | 0.536394 |
| students_dropout_and_academic_success | TabM | 0.769214 | 0.698769 | 0.879676 | 0.749653 | 0.595951 |
| website_phishing | GraphDrone | 0.907613 | 0.896814 | 0.978803 | 0.931785 | 0.245120 |
| website_phishing | TabPFN | 0.912786 | 0.898270 | 0.980122 | 0.931687 | 0.230898 |
| website_phishing | TabM | 0.849963 | 0.587882 | 0.901367 | 0.702128 | 0.439640 |
