etf_internals
==============================

etf component analytics

Project Organization
--------------------

    .
    ├── AUTHORS.md
    ├── LICENSE
    ├── README.md
    ├── bin
    ├── config
    ├── data
    │   ├── external `(downloaded prices)`
    │   ├── interim
    │   ├── processed
    │   └── raw `(holdings data)`
    ├── docs
    ├── notebooks
    ├── reports `(excel output)`
    │   └── figures
    └── src
        ├── data `(_blk_ETF_internals_)`
        ├── external
        ├── models
        ├── tools `(_blk_utilities_)`
        └── visualization

Project Information
--------------------

This project was initially developed in 2016. Some of the code 
may need to be updated. 

Please note that ETF holdings data was downloaded manually and is 
also outdated but is left here for posterity. 

This project could use a scraper for automatically downloading ETF
holdings from the issuer. The foreign symbols need to be updated in
accordance with `IEX` exchange due to the fact that, as of now `April 2018`
`Yahoo` finance api via `pandas-datareader` is deprecated.

This repo was formed using [cookiecutter-reproducible-science](https://github.com/mkrapp/cookiecutter-reproducible-science) template.
