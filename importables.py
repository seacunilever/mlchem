from mlchem.chem.manipulation import PatternRecognition as pr
from mlchem import metrics
prB = pr.Base
prR = pr.Rings
prP = pr.MolPatterns
prBn = pr.Bonds

metal_list = [
    'Li', 'Be', 'Na', 'Mg', 'Al', 'K', 'Sc', 'Ti', 'V', 'Cr',
    'Mn', 'Fe', 'Ni', 'Co', 'Cu', 'Zn', 'Ga', 'Rb', 'Sr', 'Y',
    'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In',
    'Sn', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu',
    'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
    'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am',
    'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db',
    'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv'
]

chemical_dictionary = {
    '[#B-1]': 0, '[#B]': 0, '[#Branch1]': 1, '[#Branch2]': 1, '[#Branch3]': 1,
    '[#C+1]': 0, '[#C-1]': 0, '[#C]': 1, '[#N+1]': 0, '[#N]': 1,
    '[#O+1]': 0, '[#P+1]': 0, '[#P-1]': 0, '[#P]': 0, '[#S+1]': 0, '[#S-1]': 0,
    '[#S]': 2, '[=B+1]': 0, '[=B-1]': 0, '[=B]': 1, '[=Branch1]': 10,
    '[=Branch2]': 10, '[=Branch3]': 10, '[=C+1]': 0, '[=C-1]': 0, '[=C]': 50,
    '[=N+1]': 5, '[=N-1]': 0, '[=N]': 10, '[=O+1]': 0, '[=O]': 20, '[=P+1]': 0,
    '[=P-1]': 0, '[=P]': 1, '[=Ring1]': 10, '[=Ring2]': 10, '[=Ring3]': 10,
    '[=S+1]': 0, '[=S-1]': 0, '[=S]': 2, '[B+1]': 0, '[B-1]': 0, '[B]': 1,
    '[Br]': 1, '[Branch1]': 20, '[Branch2]': 10, '[Branch3]': 10, '[C+1]': 0,
    '[C-1]': 0, '[C]': 400, '[Cl]': 1, '[F]': 1, '[H]': 30, '[I]': 1,
    '[N+1]': 4, '[N-1]': 0, '[N]': 30, '[O+1]': 0, '[O-1]': 0, '[O]': 30,
    '[P+1]': 0, '[P-1]': 0, '[P]': 1, '[Ring1]': 20, '[Ring2]': 10,
    '[Ring3]': 10, '[S+1]': 0, '[S-1]': 0, '[S]': 3
    }

colour_dictionary = {
    "maroon": (128/255, 0/255, 0/255),    # Reds
    "crimson": (220/255, 20/255, 60/255),
    "red": (255/255, 0/255, 0/255),
    "tomato": (255/255, 99/255, 71/255),
    "salmon": (250/255, 128/255, 114/255),
    "orangered": (255/255, 69/255, 0/255),
    "orange": (255/255, 165/255, 0/255),
    "gold": (255/255, 215/255, 0/255),
    "yellow": (255/255, 255/255, 0/255),
    "olive": (128/255, 128/255, 0/255),     # Greens
    "olivedrab": (107/255, 142/255, 35/255),
    "greenyellow": (173/255, 255/255, 47/255),
    "lime": (0/255, 255/255, 0/255),
    "springgreen": (0/255, 255/255, 127/255),
    "green": (0/255, 128/255, 0/255),     # Blues
    "teal": (0/255, 128/255, 128/255),
    "cyan": (0/255, 255/255, 255/255),
    "aquamarine": (127/255, 255/255, 212/255),
    "steelblue": (70/255, 130/255, 180/255),
    "cornflowerblue": (100/255, 149/255, 237/255),
    "dodgerblue": (30/255, 144/255, 255/255),
    "navy": (0/255, 0/255, 128/255),
    "blue": (0/255, 0/255, 255/255),
    "indigo": (75/255, 0/255, 130/255),     # Purples
    "slateblue": (106/255, 90/255, 205/255),
    "mediumpurple": (147/255, 112/255, 219/255),
    "darkorchid": (153/255, 50/255, 204/255),
    "purple": (128/255, 0/255, 128/255),
    "magenta": (255/255, 0/255, 255/255),
    "orchid": (218/255, 112/255, 214/255),
    "mediumvioletred": (199/255, 21/255, 133/255),
    "deeppink": (255/255, 20/255, 147/255),
    "hotpink": (255/255, 105/255, 180/255),
    "pink": (255/255, 192/255, 203/255),
    "lavender": (230/255, 230/255, 250/255),
    "beige": (245/255, 245/255, 220/255),     # Browns
    "saddlebrown": (139/255, 69/255, 19/255),
    "sienna": (160/255, 82/255, 45/255),
    "peru": (205/255, 133/255, 63/255),
    "black": (0/255, 0/255, 0/255),     # White, Black, Grey
    "grey": (128/255, 128/255, 128/255),
    "white": (255/255, 255/255, 255/255),
}

chemotype_dictionary = {

    # Patterns by element #

    # C
    'Carbon': [prP.check_carbon, {}],
    'Carbanion': [prP.check_carbanion, {}],
    'Carbocation': [prP.check_carbocation, {}],
    'Carbon > 40% tot atoms': [prB.pattern_abs_fraction_greater_than,
                               {'func': prP.check_carbon,
                                'threshold': 0.4}
                               ],
    'Carbon > 60% tot atoms': [prB.pattern_abs_fraction_greater_than,
                               {'func': prP.check_carbon,
                                'threshold': 0.6}
                               ],
    'Carbon > 80% tot atoms': [prB.pattern_abs_fraction_greater_than,
                               {'func': prP.check_carbon,
                                'threshold': 0.8}
                               ],

    'Aromatic Carbon': [prP.check_pattern_aromatic,
                        {'pattern_function': prP.check_carbon}
                        ],
    'Aromatic Carbon > 10% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                        {'func': prP.check_pattern_aromatic,
                                         'threshold': 0.1,
                                         'hidden_pattern_function': prP.
                                            check_carbon}
                                        ],
    'Aromatic Carbon > 30% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                        {'func': prP.check_pattern_aromatic,
                                         'threshold': 0.3,
                                         'hidden_pattern_function': prP.
                                            check_carbon}
                                        ],
    'Aromatic Carbon > 50% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                        {'func': prP.check_pattern_aromatic,
                                         'threshold': 0.5,
                                         'hidden_pattern_function': prP.
                                            check_carbon}
                                        ],

    'Aliphatic Carbon': [prP.check_pattern_aliphatic,
                         {'pattern_function': prP.check_carbon}
                         ],
    'Aliphatic Carbon > 10% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                         {'func': prP.check_pattern_aliphatic,
                                          'threshold': 0.1,
                                          'hidden_pattern_function': prP.
                                          check_carbon}
                                         ],
    'Aliphatic Carbon > 30% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                         {'func': prP.check_pattern_aliphatic,
                                          'threshold': 0.3,
                                          'hidden_pattern_function': prP.
                                          check_carbon}
                                         ],
    'Aliphatic Carbon > 50% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                         {'func': prP.check_pattern_aliphatic,
                                          'threshold': 0.5,
                                          'hidden_pattern_function': prP.
                                          check_carbon}
                                         ],

    'Cyclic Carbon': [prR.check_pattern_cyclic,
                      {'pattern_function': prP.check_carbon}
                      ],
    'Cyclic Carbon > 10% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                      {'func': prR.check_pattern_cyclic,
                                       'threshold': 0.1,
                                       'hidden_pattern_function': prP.
                                       check_carbon}
                                      ],
    'Cyclic Carbon > 30% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                      {'func': prR.check_pattern_cyclic,
                                       'threshold': 0.3,
                                       'hidden_pattern_function': prP.
                                       check_carbon}
                                      ],
    'Cyclic Carbon > 50% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                      {'func': prR.check_pattern_cyclic,
                                       'threshold': 0.5,
                                       'hidden_pattern_function': prP.
                                       check_carbon}
                                      ],

    'Alkyl Carbon': [prP.check_alkyl_carbon, {}],
    'Alkyl Carbon > 10% tot C': [prB.pattern_rel_fraction_greater_than,
                                 {'func1': prP.check_alkyl_carbon,
                                  'func2': prP.check_carbon,
                                  'threshold': 0.1}
                                 ],
    'Alkyl Carbon > 30% tot C': [prB.pattern_rel_fraction_greater_than,
                                 {'func1': prP.check_alkyl_carbon,
                                  'func2': prP.check_carbon,
                                  'threshold': 0.3}
                                 ],
    'Alkyl Carbon > 50% tot C': [prB.pattern_rel_fraction_greater_than,
                                 {'func1': prP.check_alkyl_carbon,
                                  'func2': prP.check_carbon,
                                  'threshold': 0.5}
                                 ],

    'Allenic Carbon': [prP.check_allenic_carbon, {}],
    'Allenic Carbon > 5% tot C': [prB.pattern_rel_fraction_greater_than,
                                  {'func1': prP.check_allenic_carbon,
                                   'func2': prP.check_carbon,
                                   'threshold': 0.05}
                                  ],
    'Allenic Carbon > 10% tot C': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_allenic_carbon,
                                    'func2': prP.check_carbon,
                                    'threshold': 0.1}
                                   ],

    'Vinylic Carbon': [prP.check_vinylic_carbon, {}],
    'Vinylic Carbon > 5% tot C': [prB.pattern_rel_fraction_greater_than,
                                  {'func1': prP.check_vinylic_carbon,
                                   'func2': prP.check_carbon,
                                   'threshold': 0.05}
                                  ],
    'Vinylic Carbon > 10% tot C': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_vinylic_carbon,
                                    'func2': prP.check_carbon,
                                    'threshold': 0.1}
                                   ],

    'Acetylenic Carbon': [prP.check_acetylenic_carbon, {}],
    'Acetylenic Carbon > 5% tot C': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prP.check_acetylenic_carbon,
                                      'func2': prP.check_carbon,
                                      'threshold': 0.05}
                                     ],
    'Acetylenic Carbon > 10% tot C': [prB.pattern_rel_fraction_greater_than,
                                      {'func1': prP.check_acetylenic_carbon,
                                       'func2': prP.check_carbon,
                                       'threshold': 0.1}
                                      ],

    # C, O

    'Carbonyl': [prP.check_carbonyl, {}],
    'Carbonyl-Aryl': [prP.check_pattern_aromatic_substituent,
                      {'pattern_function': prP.check_carbonyl}
                      ],
    'Carbonyl-Ring': [prR.check_pattern_cyclic_substituent,
                      {'pattern_function': prP.check_carbonyl}
                      ],

    '1,2-dicarbonyl': [prP.check_alpha_dicarbonyl, {}],
    '1,2-dicarbonyl (Aliphatic)': [prP.check_pattern_aliphatic,
                                   {'pattern_function': prP.
                                    check_alpha_dicarbonyl}
                                   ],
    '1,2-dicarbonyl (Aromatic)': [prP.check_pattern_aromatic,
                                  {'pattern_function': prP.
                                   check_alpha_dicarbonyl}
                                  ],
    '1,2-dicarbonyl (Cyclic)': [prR.check_pattern_cyclic,
                                {'pattern_function': prP.
                                 check_alpha_dicarbonyl}
                                ],

    '1,3-dicarbonyl': [prP.check_beta_dicarbonyl, {}],
    '1,3-dicarbonyl (Aliphatic)': [prP.check_pattern_aliphatic,
                                   {'pattern_function': prP.
                                    check_beta_dicarbonyl}
                                   ],
    '1,3-dicarbonyl (Aromatic)': [prP.check_pattern_aromatic,
                                  {'pattern_function': prP.
                                   check_beta_dicarbonyl}
                                  ],
    '1,3-dicarbonyl (Cyclic)': [prR.check_pattern_cyclic,
                                {'pattern_function': prP.check_beta_dicarbonyl}
                                ],

    '1,4-dicarbonyl': [prP.check_gamma_dicarbonyl, {}],
    '1,4-dicarbonyl (Aliphatic)': [prP.check_pattern_aliphatic,
                                   {'pattern_function': prP.
                                    check_gamma_dicarbonyl}
                                   ],
    '1,4-dicarbonyl (Aromatic)': [prP.check_pattern_aromatic,
                                  {'pattern_function': prP.
                                   check_gamma_dicarbonyl}
                                  ],
    '1,4-dicarbonyl (Cyclic)': [prR.check_pattern_cyclic,
                                {'pattern_function': prP.
                                 check_gamma_dicarbonyl}
                                ],

    '1,5-dicarbonyl': [prP.check_delta_dicarbonyl, {}],
    '1,5-dicarbonyl (Aliphatic)': [prP.check_pattern_aliphatic,
                                   {'pattern_function': prP.
                                    check_delta_dicarbonyl}
                                   ],
    '1,5-dicarbonyl (Aromatic)': [prP.check_pattern_aromatic,
                                  {'pattern_function': prP.
                                   check_delta_dicarbonyl}
                                  ],
    '1,5-dicarbonyl (Cyclic)': [prR.check_pattern_cyclic,
                                {'pattern_function': prP.
                                 check_delta_dicarbonyl}
                                ],

    'Acyl Halide': [prP.check_acyl_halide, {}],
    'Acyl Halide (Cyclic)': [prR.check_pattern_cyclic,
                             {'pattern_function': prP.check_acyl_halide}
                             ],
    'Acyl Halide-Aryl': [prP.check_pattern_aromatic_substituent,
                         {'pattern_function': prP.check_acyl_halide}
                         ],
    'Acyl Halide-Ring': [prR.check_pattern_cyclic_substituent,
                         {'pattern_function': prP.check_acyl_halide}
                         ],
    'Acyl Halide > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                                  {'func1': prP.check_acyl_halide,
                                   'func2': prP.check_carbonyl,
                                   'threshold': 0.5}
                                  ],

    'Aldehyde': [prP.check_aldehyde, {}],
    'Aldehyde (Aromatic)': [prP.check_pattern_aromatic,
                            {'pattern_function': prP.check_aldehyde}
                            ],
    'Aldehyde (Cyclic)': [prR.check_pattern_cyclic,
                          {'pattern_function': prP.check_aldehyde}
                          ],
    'Aldehyde-Aryl': [prP.check_pattern_aromatic_substituent,
                      {'pattern_function': prP.check_aldehyde}
                      ],
    'Aldehyde-Ring': [prR.check_pattern_cyclic_substituent,
                      {'pattern_function': prP.check_aldehyde}
                      ],
    'Aldehyde > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                               {'func1': prP.check_aldehyde,
                                'func2': prP.check_carbonyl,
                                'threshold': 0.5}
                               ],
    'Aldehyde > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                             {'func1': prP.check_aldehyde,
                              'func2': prP.check_oxygen,
                              'threshold': 0.5}
                             ],

    'Anhydride': [prP.check_anhydride, {}],
    'Anhydride (Aromatic)': [prP.check_pattern_aromatic,
                             {'pattern_function': prP.check_anhydride}
                             ],
    'Anhydride (Cyclic)': [prR.check_pattern_cyclic,
                           {'pattern_function': prP.check_anhydride}
                           ],
    'Anhydride-Aryl': [prP.check_pattern_aromatic_substituent,
                       {'pattern_function': prP.check_anhydride}
                       ],
    'Anhydride-Ring': [prR.check_pattern_cyclic_substituent,
                       {'pattern_function': prP.check_anhydride}
                       ],
    'Anhydride > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                                {'func1': prP.check_anhydride,
                                 'func2': prP.check_carbonyl,
                                 'threshold': 0.5}
                                ],
    'Anhydride > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                              {'func1': prP.check_anhydride,
                               'func2': prP.check_oxygen,
                               'threshold': 0.5}
                              ],

    'Carboxyl': [prP.check_carboxyl, {}],
    'Carboxyl-Aryl': [prP.check_pattern_aromatic_substituent,
                      {'pattern_function': prP.check_carboxyl}],
    'Carboxyl-Ring': [prR.check_pattern_cyclic_substituent,
                      {'pattern_function': prP.check_carboxyl}
                      ],
    'Carboxyl > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                               {'func1': prP.check_carboxyl,
                                'func2': prP.check_carbonyl,
                                'threshold': 0.5}
                               ],
    'Carboxyl > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                             {'func1': prP.check_carboxyl,
                              'func2': prP.check_oxygen,
                              'threshold': 0.5}
                             ],

    'Carbonic Acid': [prP.check_carbonic_acid, {}],
    'Carbonic Acid > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_carbonic_acid,
                                     'func2': prP.check_carbonyl,
                                     'threshold': 0.5}
                                    ],

    'Carbonate Ester': [prP.check_carbonate_ester, {}],
    'Carbonate Ester (Aromatic)': [prP.check_pattern_aromatic,
                                   {'pattern_function': prP.check_carbonate_ester}
                                   ],
    'Carbonate Ester (Cyclic)': [prR.check_pattern_cyclic,
                                 {'pattern_function': prP.check_carbonate_ester}
                                 ],
    'Carbonate Ester-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_carbonate_ester}
                             ],
    'Carbonate Ester-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_carbonate_ester}
                             ],
    'Carbonate Ester > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                                      {'func1': prP.check_carbonate_ester,
                                       'func2': prP.check_carbonyl,
                                       'threshold': 0.5}
                                      ],
    'Carbonate Ester > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_carbonate_ester,
                                     'func2': prP.check_oxygen,
                                     'threshold': 0.5}
                                    ],

    'Ester': [prP.check_ester, {}],
    'Ester (Aliphatic)': [prP.check_pattern_aliphatic,
                          {'pattern_function': prP.check_ester}
                          ],
    'Ester (Aromatic)': [prP.check_pattern_aromatic,
                         {'pattern_function': prP.check_ester}
                         ],
    'Ester (Cyclic)': [prR.check_pattern_cyclic,
                       {'pattern_function': prP.check_ester}
                       ],
    'Ester-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_ester}
                   ],
    'Ester-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_ester}
                   ],
    'Ester > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_ester,
                             'func2': prP.check_carbonyl,
                             'threshold': 0.5}
                            ],
    'Ester > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_ester,
                           'func2': prP.check_oxygen,
                           'threshold': 0.5}
                          ],

    'Ketone': [prP.check_ketone, {}],
    'Ketone (Aliphatic)': [prP.check_pattern_aliphatic,
                           {'pattern_function': prP.check_ketone}
                           ],
    'Ketone (Aromatic)': [prP.check_pattern_aromatic,
                          {'pattern_function': prP.check_ketone}
                          ],
    'Ketone (Cyclic)': [prR.check_pattern_cyclic,
                        {'pattern_function': prP.check_ketone}
                        ],
    'Ketone-Aryl': [prP.check_pattern_aromatic_substituent,
                    {'pattern_function': prP.check_ketone}
                    ],
    'Ketone-Ring': [prR.check_pattern_cyclic_substituent,
                    {'pattern_function': prP.check_ketone}
                    ],
    'Ketone > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                             {'func1': prP.check_ketone,
                              'func2': prP.check_carbonyl,
                              'threshold': 0.5}
                             ],
    'Ketone > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                           {'func1': prP.check_ketone,
                            'func2': prP.check_oxygen,
                            'threshold': 0.5}
                           ],

    '1,2-diketone': [prP.check_alpha_diketone, {}],
    '1,3-diketone': [prP.check_beta_diketone, {}],
    '1,4-diketone': [prP.check_gamma_diketone, {}],

    'Ether': [prP.check_ether, {}],
    'Ether (Aliphatic)': [prP.check_pattern_aliphatic,
                          {'pattern_function': prP.check_ether}
                          ],
    'Ether (Aromatic)': [prP.check_pattern_aromatic,
                         {'pattern_function': prP.check_ether}
                         ],
    'Ether (Cyclic)': [prR.check_pattern_cyclic,
                       {'pattern_function': prP.check_ether}
                       ],
    'Ether-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_ether}
                   ],
    'Ether-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_ether}
                   ],
    'Ether > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_ether,
                           'func2': prP.check_oxygen,
                           'threshold': 0.5}
                          ],

    'Amide': [prP.check_amide, {}],
    'Amide (Aliphatic)': [prP.check_pattern_aliphatic,
                          {'pattern_function': prP.check_amide}
                          ],
    'Amide (Aromatic)': [prP.check_pattern_aromatic,
                         {'pattern_function': prP.check_amide}
                         ],
    'Amide (Cyclic)': [prR.check_pattern_cyclic,
                       {'pattern_function': prP.check_amide}
                       ],
    'Amide-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_amide}
                   ],
    'Amide-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_amide}
                   ],
    'Amide > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_amide,
                             'func2': prP.check_carbonyl,
                             'threshold': 0.5}
                            ],
    'Amide > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_amide,
                           'func2': prP.check_nitrogen,
                           'threshold': 0.5}
                          ],

    'Carbamate': [prP.check_carbamate, {}],
    'Carbamate (Aromatic)': [prP.check_pattern_aromatic,
                             {'pattern_function': prP.check_carbamate}
                             ],
    'Carbamate (Cyclic)': [prR.check_pattern_cyclic,
                           {'pattern_function': prP.check_carbamate}
                           ],
    'Carbamate-Aryl': [prP.check_pattern_aromatic_substituent,
                       {'pattern_function': prP.check_carbamate}
                       ],
    'Carbamate-Ring': [prR.check_pattern_cyclic_substituent,
                       {'pattern_function': prP.check_carbamate}
                       ],
    'Carbamate > 50% tot C=O': [prB.pattern_rel_fraction_greater_than,
                                {'func1': prP.check_carbamate,
                                 'func2': prP.check_carbonyl,
                                 'threshold': 0.5}
                                ],
    'Carbamate > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                              {'func1': prP.check_carbamate,
                               'func2': prP.check_nitrogen,
                               'threshold': 0.5}
                              ],

    # N

    'Nitrogen': [prP.check_nitrogen, {}],
    'Nitrogen > 5% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                {'func': prP.check_nitrogen,
                                 'threshold': 0.05}
                                ],
    'Nitrogen > 10% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                 {'func': prP.check_nitrogen,
                                  'threshold': 0.1}
                                 ],
    'Nitrogen > 25% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                 {'func': prP.check_nitrogen,
                                  'threshold': 0.25}
                                 ],

    'Aliphatic Nitrogen': [prP.check_pattern_aliphatic,
                           {'pattern_function': prP.check_nitrogen}
                           ],
    'Aromatic Nitrogen': [prP.check_pattern_aromatic,
                          {'pattern_function': prP.check_nitrogen}
                          ],
    'Cyclic Nitrogen': [prR.check_pattern_cyclic,
                        {'pattern_function': prP.check_nitrogen}
                        ],

    'Amine': [prP.check_amine, {}],
    'Amine (Aliphatic)': [prP.check_pattern_aliphatic,
                          {'pattern_function': prP.check_amine}
                          ],
    'Amine (Aromatic)': [prP.check_pattern_aromatic,
                         {'pattern_function': prP.check_amine}
                         ],
    'Amine (Cyclic)': [prR.check_pattern_cyclic,
                       {'pattern_function': prP.check_amine}
                       ],
    'Amine-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_amine}
                   ],
    'Amine-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_amine}
                   ],
    'Primary Amine': [prP.check_amine_primary, {}],
    'Primary Amine > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                                  {'func1': prP.check_amine_primary,
                                   'func2': prP.check_nitrogen,
                                   'threshold': 0.5}
                                  ],
    'Secondary Amine': [prP.check_amine_secondary, {}],
    'Secondary Amine > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_amine_secondary,
                                     'func2': prP.check_nitrogen,
                                     'threshold': 0.5}
                                    ],
    'Tertiary Amine': [prP.check_amine_tertiary, {}],
    'Tertiary Amine > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_amine_tertiary,
                                    'func2': prP.check_nitrogen,
                                    'threshold': 0.5}
                                   ],
    'Quaternary Amine': [prP.check_amine_quaternary, {}],
    'Quaternary Amine > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prP.check_amine_quaternary,
                                      'func2': prP.check_nitrogen,
                                      'threshold': 0.5}
                                     ],
    'Enamine': [prP.check_enamine, {}],

    'Aminoacid': [prP.check_aminoacid, {}],
    'Alanine': [prP.check_alanine, {}],
    'Arginine': [prP.check_arginine, {}],
    'Asparagine': [prP.check_asparagine, {}],
    'Aspartate': [prP.check_aspartate, {}],
    'Cysteine': [prP.check_cysteine, {}],
    'Glutamate': [prP.check_glutamate, {}],
    'Glycine': [prP.check_glycine, {}],
    'Histidine': [prP.check_histidine, {}],
    'Isoleucine': [prP.check_isoleucine, {}],
    'Leucine': [prP.check_leucine, {}],
    'Lysine': [prP.check_lysine, {}],
    'Methionine': [prP.check_methionine, {}],
    'Phenylalanine': [prP.check_phenylalanine, {}],
    'Proline': [prP.check_proline, {}],
    'Serine': [prP.check_serine, {}],
    'Threonine': [prP.check_threonine, {}],
    'Tryptophan': [prP.check_tryptophan, {}],
    'Tyrosine': [prP.check_tyrosine, {}],
    'Valine': [prP.check_valine, {}],

    'Azide': [prP.check_azide, {}],
    'Azide-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_azide}
                   ],
    'Azide-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_azide}
                   ],
    'Azide > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_azide,
                           'func2': prP.check_nitrogen,
                           'threshold': 0.5}
                          ],

    'Azo': [prP.check_azo, {}],
    'Azo-Aryl': [prP.check_pattern_aromatic_substituent,
                 {'pattern_function': prP.check_azo}
                 ],
    'Azo-Ring': [prR.check_pattern_cyclic_substituent,
                 {'pattern_function': prP.check_azo}
                 ],
    'Azo > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                        {'func1': prP.check_azo,
                         'func2': prP.check_nitrogen,
                         'threshold': 0.5}
                        ],

    'Azoxy': [prP.check_azoxy, {}],
    'Azoxy-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_azoxy}
                   ],
    'Azoxy-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_azoxy}
                   ],
    'Azoxy > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_azoxy,
                           'func2': prP.check_nitrogen,
                           'threshold': 0.5}
                          ],

    'Diazo': [prP.check_diazo, {}],
    'Diazo-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_diazo}
                   ],
    'Diazo-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_diazo}
                   ],
    'Diazo > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_diazo,
                           'func2': prP.check_nitrogen,
                           'threshold': 0.5}
                          ],

    'Hydrazine': [prP.check_hydrazine, {}],
    'Hydrazine (Aromatic)': [prP.check_pattern_aromatic,
                             {'pattern_function': prP.check_hydrazine}
                             ],
    'Hydrazine (Cyclic)': [prR.check_pattern_cyclic,
                           {'pattern_function': prP.check_hydrazine}
                           ],
    'Hydrazine-Aryl': [prP.check_pattern_aromatic_substituent,
                       {'pattern_function': prP.check_hydrazine}
                       ],
    'Hydrazine-Ring': [prR.check_pattern_cyclic_substituent,
                       {'pattern_function': prP.check_hydrazine}
                       ],
    'Hydrazine > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                              {'func1': prP.check_hydrazine,
                               'func2': prP.check_nitrogen,
                               'threshold': 0.5}
                              ],

    'Hydrazone': [prP.check_hydrazone, {}],
    'Hydrazone (Aromatic)': [prP.check_pattern_aromatic,
                             {'pattern_function': prP.check_hydrazone}
                             ],
    'Hydrazone (Cyclic)': [prR.check_pattern_cyclic,
                           {'pattern_function': prP.check_hydrazone}
                           ],
    'Hydrazone-Aryl': [prP.check_pattern_aromatic_substituent,
                       {'pattern_function': prP.check_hydrazone}
                       ],
    'Hydrazone-Ring': [prR.check_pattern_cyclic_substituent,
                       {'pattern_function': prP.check_hydrazone}
                       ],
    'Hydrazone > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                              {'func1': prP.check_hydrazone,
                               'func2': prP.check_nitrogen,
                               'threshold': 0.5}
                              ],

    'Imine': [prP.check_imine, {}],
    'Imine (Aromatic)': [prP.check_pattern_aromatic,
                         {'pattern_function': prP.check_imine}
                         ],
    'Imine (Cyclic)': [prR.check_pattern_cyclic,
                       {'pattern_function': prP.check_imine}
                       ],
    'Imine-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_imine}
                   ],
    'Imine-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_imine}
                   ],
    'Imine > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_imine,
                           'func2': prP.check_nitrogen,
                           'threshold': 0.5}
                          ],
    'Iminium': [prP.check_iminium, {}],

    'Imide': [prP.check_imide, {}],
    'Imide (Aromatic)': [prP.check_pattern_aromatic,
                         {'pattern_function': prP.check_imide}
                         ],
    'Imide (Cyclic)': [prR.check_pattern_cyclic,
                       {'pattern_function': prP.check_imide}
                       ],
    'Imide-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_imide}
                   ],
    'Imide-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_imide}
                   ],
    'Imide > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_imide,
                           'func2': prP.check_nitrogen,
                           'threshold': 0.5}
                          ],

    'Nitrate': [prP.check_nitrate, {}],
    'Nitrate-Aryl': [prP.check_pattern_aromatic_substituent,
                     {'pattern_function': prP.check_nitrate}
                     ],
    'Nitrate-Ring': [prR.check_pattern_cyclic_substituent,
                     {'pattern_function': prP.check_nitrate}
                     ],
    'Nitrate > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_nitrate,
                             'func2': prP.check_nitrogen,
                             'threshold': 0.5}
                            ],

    'Nitro': [prP.check_nitro, {}],
    'Nitro-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_nitro}
                   ],
    'Nitro-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_nitro}
                   ],
    'Nitro > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_nitro,
                           'func2': prP.check_nitrogen,
                           'threshold': 0.5}
                          ],

    'Nitrile': [prP.check_nitrile, {}],
    'Nitrile-Aryl': [prP.check_pattern_aromatic_substituent,
                     {'pattern_function': prP.check_nitrile}
                     ],
    'Nitrile-Ring': [prR.check_pattern_cyclic_substituent,
                     {'pattern_function': prP.check_nitrile}
                     ],
    'Nitrile > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_nitrile,
                             'func2': prP.check_nitrogen,
                             'threshold': 0.5}
                            ],

    'Isonitrile': [prP.check_isonitrile, {}],
    'Isonitrile-Aryl': [prP.check_pattern_aromatic_substituent,
                        {'pattern_function': prP.check_isonitrile}
                        ],
    'Isonitrile-Ring': [prR.check_pattern_cyclic_substituent,
                        {'pattern_function': prP.check_isonitrile}
                        ],
    'Isonitrile > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                               {'func1': prP.check_isonitrile,
                                'func2': prP.check_nitrogen,
                                'threshold': 0.5}
                               ],

    'Nitroso': [prP.check_nitroso, {}],
    'Nitroso-Aryl': [prP.check_pattern_aromatic_substituent,
                     {'pattern_function': prP.check_nitroso}
                     ],
    'Nitroso-Ring': [prR.check_pattern_cyclic_substituent,
                     {'pattern_function': prP.check_nitroso}
                     ],
    'Nitroso > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_nitroso,
                             'func2': prP.check_nitrogen,
                             'threshold': 0.5}
                            ],

    'N-oxide': [prP.check_n_oxide, {}],
    'N-oxide (Aromatic)': [prP.check_pattern_aromatic,
                           {'pattern_function': prP.check_n_oxide}
                           ],
    'N-oxide (Cyclic)': [prR.check_pattern_cyclic,
                         {'pattern_function': prP.check_n_oxide}
                         ],
    'N-oxide-Aryl': [prP.check_pattern_aromatic_substituent,
                     {'pattern_function': prP.check_n_oxide}
                     ],
    'N-oxide-Ring': [prR.check_pattern_cyclic_substituent,
                     {'pattern_function': prP.check_n_oxide}
                     ],
    'N-oxide > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_n_oxide,
                             'func2': prP.check_nitrogen,
                             'threshold': 0.5}
                            ],

    'Cyanamide': [prP.check_cyanamide, {}],
    'Cyanamide (Aromatic)': [prP.check_pattern_aromatic,
                             {'pattern_function': prP.check_cyanamide}
                             ],
    'Cyanamide (Cyclic)': [prR.check_pattern_cyclic,
                           {'pattern_function': prP.check_cyanamide}
                           ],
    'Cyanamide-Aryl': [prP.check_pattern_aromatic_substituent,
                       {'pattern_function': prP.check_cyanamide}
                       ],
    'Cyanamide-Ring': [prR.check_pattern_cyclic_substituent,
                       {'pattern_function': prP.check_cyanamide}
                       ],
    'Cyanamide > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                              {'func1': prP.check_cyanamide,
                               'func2': prP.check_nitrogen,
                               'threshold': 0.5}
                              ],

    'Cyanate': [prP.check_cyanate, {}],
    'Cyanate (Aromatic)': [prP.check_pattern_aromatic,
                           {'pattern_function': prP.check_cyanate}
                           ],
    'Cyanate (Cyclic)': [prR.check_pattern_cyclic,
                         {'pattern_function': prP.check_cyanate}
                         ],
    'Cyanate-Aryl': [prP.check_pattern_aromatic_substituent,
                     {'pattern_function': prP.check_cyanate}
                     ],
    'Cyanate-Ring': [prR.check_pattern_cyclic_substituent,
                     {'pattern_function': prP.check_cyanate}
                     ],
    'Cyanate > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_cyanate,
                             'func2': prP.check_nitrogen,
                             'threshold': 0.5}
                            ],

    'Isocyanate': [prP.check_isocyanate, {}],
    'Isocyanate (Aromatic)': [prP.check_pattern_aromatic,
                              {'pattern_function': prP.check_isocyanate}
                              ],
    'Isocyanate (Cyclic)': [prR.check_pattern_cyclic,
                            {'pattern_function': prP.check_isocyanate}
                            ],
    'Isocyanate-Aryl': [prP.check_pattern_aromatic_substituent,
                        {'pattern_function': prP.check_isocyanate}
                        ],
    'Isocyanate-Ring': [prR.check_pattern_cyclic_substituent,
                        {'pattern_function': prP.check_isocyanate}
                        ],
    'Isocyanate > 50% tot N': [prB.pattern_rel_fraction_greater_than,
                               {'func1': prP.check_isocyanate,
                                'func2': prP.check_nitrogen,
                                'threshold': 0.5}
                               ],

    # O

    'Oxygen': [prP.check_oxygen, {}],
    'Oxygen > 10% tot atoms': [prB.pattern_abs_fraction_greater_than,
                               {'func': prP.check_oxygen,
                                'threshold': 0.1}
                               ],
    'Oxygen > 20% tot atoms': [prB.pattern_abs_fraction_greater_than,
                               {'func': prP.check_oxygen,
                                'threshold': 0.2}
                               ],
    'Oxygen > 30% tot atoms': [prB.pattern_abs_fraction_greater_than,
                               {'func': prP.check_oxygen,
                                'threshold': 0.3}
                               ],
    'Alcohol': [prP.check_alcohol, {}],
    'Alcohol (Aromatic)': [prP.check_pattern_aromatic,
                           {'pattern_function': prP.check_alcohol}
                           ],
    'Alcohol (Cyclic)': [prR.check_pattern_cyclic,
                         {'pattern_function': prP.check_alcohol}
                         ],
    'Alcohol-Aryl': [prP.check_pattern_aromatic_substituent,
                     {'pattern_function': prP.check_alcohol}
                     ],
    'Alcohol-Ring': [prR.check_pattern_cyclic_substituent,
                     {'pattern_function': prP.check_alcohol}
                     ],
    'Alcohol > 50% tot O': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_alcohol,
                             'func2': prP.check_oxygen,
                             'threshold': 0.5}
                            ],

    'Enol': [prP.check_enol, {}],

    # P

    'Phosphorus': [prP.check_phosphorus, {}],
    'Phosphorus > 5% tot atoms': [prB.pattern_abs_fraction_greater_than,
                                  {'func': prP.check_phosphorus,
                                   'threshold': 0.05}
                                  ],

    'Phosphoric Acid': [prP.check_phosphoric_acid, {}],
    'Phosphoric Acid-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_phosphoric_acid}
                             ],
    'Phosphoric Acid-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_phosphoric_acid}
                             ],

    'Phosphoric Ester': [prP.check_phosphoric_ester, {}],
    'Phosphoric Ester-Aryl': [prP.check_pattern_aromatic_substituent,
                              {'pattern_function': prP.check_phosphoric_ester}
                              ],
    'Phosphoric Ester-Ring': [prR.check_pattern_cyclic_substituent,
                              {'pattern_function': prP.check_phosphoric_ester}
                              ],

    # S

    'Sulphur': [prP.check_sulphur, {}],
    'Sulphur > 5% tot atoms': [prB.pattern_abs_fraction_greater_than,
                               {'func': prP.check_sulphur,
                                'threshold': 0.05}
                               ],

    'Thiol': [prP.check_thiol, {}],
    'Thiol (Aromatic)': [prP.check_pattern_aromatic,
                         {'pattern_function': prP.check_thiol}
                         ],
    'Thiol (Cyclic)': [prR.check_pattern_cyclic,
                       {'pattern_function': prP.check_thiol}
                       ],
    'Thiol-Aryl': [prP.check_pattern_aromatic_substituent,
                   {'pattern_function': prP.check_thiol}
                   ],
    'Thiol-Ring': [prR.check_pattern_cyclic_substituent,
                   {'pattern_function': prP.check_thiol}
                   ],
    'Thiol > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                          {'func1': prP.check_thiol,
                           'func2': prP.check_sulphur,
                           'threshold': 0.5}
                          ],

    'Thiocarbonyl': [prP.check_thiocarbonyl, {}],
    'Thiocarbonyl (Aromatic)': [prP.check_pattern_aromatic,
                                {'pattern_function': prP.check_thiocarbonyl}
                                ],
    'Thiocarbonyl (Cyclic)': [prR.check_pattern_cyclic,
                              {'pattern_function': prP.check_thiocarbonyl}
                              ],
    'Thiocarbonyl-Aryl': [prP.check_pattern_aromatic_substituent,
                          {'pattern_function': prP.check_thiocarbonyl}
                          ],
    'Thiocarbonyl-Ring': [prR.check_pattern_cyclic_substituent,
                          {'pattern_function': prP.check_thiocarbonyl}
                          ],
    'Thiocarbonyl > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                 {'func1': prP.check_thiocarbonyl,
                                  'func2': prP.check_sulphur,
                                  'threshold': 0.5}
                                 ],

    'Thioketone': [prP.check_thioketone, {}],
    'Thioketone (Aromatic)': [prP.check_pattern_aromatic,
                              {'pattern_function': prP.check_thioketone}
                              ],
    'Thioketone (Cyclic)': [prR.check_pattern_cyclic,
                            {'pattern_function': prP.check_thioketone}
                            ],
    'Thioketone-Aryl': [prP.check_pattern_aromatic_substituent,
                        {'pattern_function': prP.check_thioketone}
                        ],
    'Thioketone-Ring': [prR.check_pattern_cyclic_substituent,
                        {'pattern_function': prP.check_thioketone}
                        ],
    'Thioketone > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                               {'func1': prP.check_thioketone,
                                'func2': prP.check_sulphur,
                                'threshold': 0.5}
                               ],

    'Thioaldehyde': [prP.check_thioaldehyde, {}],
    'Thioaldehyde (Aromatic)': [prP.check_pattern_aromatic,
                                {'pattern_function': prP.check_thioaldehyde}
                                ],
    'Thioaldehyde (Cyclic)': [prR.check_pattern_cyclic,
                              {'pattern_function': prP.check_thioaldehyde}
                              ],
    'Thioaldehyde-Aryl': [prP.check_pattern_aromatic_substituent,
                          {'pattern_function': prP.check_thioaldehyde}
                          ],
    'Thioaldehyde-Ring': [prR.check_pattern_cyclic_substituent,
                          {'pattern_function': prP.check_thioaldehyde}
                          ],
    'Thioaldehyde > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                 {'func1': prP.check_thioaldehyde,
                                  'func2': prP.check_sulphur,
                                  'threshold': 0.5}
                                 ],

    'Thioanhydride': [prP.check_thioanhydride, {}],
    'Thioanhydride (Aromatic)': [prP.check_pattern_aromatic,
                                 {'pattern_function': prP.check_thioanhydride}
                                 ],
    'Thioanhydride (Cyclic)': [prR.check_pattern_cyclic,
                               {'pattern_function': prP.check_thioanhydride}
                               ],
    'Thioanhydride-Aryl': [prP.check_pattern_aromatic_substituent,
                           {'pattern_function': prP.check_thioanhydride}
                           ],
    'Thioanhydride-Ring': [prR.check_pattern_cyclic_substituent,
                           {'pattern_function': prP.check_thioanhydride}
                           ],
    'Thioanhydride > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                  {'func1': prP.check_thioanhydride,
                                   'func2': prP.check_sulphur,
                                   'threshold': 0.5}
                                  ],

    'Thiocarboxylic': [prP.check_thiocarboxylic, {}],
    'Thiocarboxylic (Aromatic)': [prP.check_pattern_aromatic,
                                  {'pattern_function': prP.check_thiocarboxylic}
                                  ],
    'Thiocarboxylic (Cyclic)': [prR.check_pattern_cyclic,
                                {'pattern_function': prP.check_thiocarboxylic}
                                ],
    'Thiocarboxylic-Aryl': [prP.check_pattern_aromatic_substituent,
                            {'pattern_function': prP.check_thiocarboxylic}
                            ],
    'Thiocarboxylic-Ring': [prR.check_pattern_cyclic_substituent,
                            {'pattern_function': prP.check_thiocarboxylic}
                            ],
    'Thiocarboxylic > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_thiocarboxylic,
                                    'func2': prP.check_sulphur,
                                    'threshold': 0.5}
                                   ],

    'Thioester': [prP.check_thioester, {}],
    'Thioester (Aromatic)': [prP.check_pattern_aromatic,
                             {'pattern_function': prP.check_thioester}
                             ],
    'Thioester (Cyclic)': [prR.check_pattern_cyclic,
                           {'pattern_function': prP.check_thioester}
                           ],
    'Thioester-Aryl': [prP.check_pattern_aromatic_substituent,
                       {'pattern_function': prP.check_thioester}
                       ],
    'Thioester-Ring': [prR.check_pattern_cyclic_substituent,
                       {'pattern_function': prP.check_thioester}
                       ],
    'Thioester > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                              {'func1': prP.check_thioester,
                               'func2': prP.check_sulphur,
                               'threshold': 0.5}
                              ],

    'Sulphide': [prP.check_sulphide, {}],
    'Sulphide (Aromatic)': [prP.check_pattern_aromatic,
                            {'pattern_function': prP.check_sulphide}
                            ],
    'Sulphide (Cyclic)': [prR.check_pattern_cyclic,
                          {'pattern_function': prP.check_sulphide}
                          ],
    'Sulphide-Aryl': [prP.check_pattern_aromatic_substituent,
                      {'pattern_function': prP.check_sulphide}
                      ],
    'Sulphide-Ring': [prR.check_pattern_cyclic_substituent,
                      {'pattern_function': prP.check_sulphide}
                      ],
    'Sulphide > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                             {'func1': prP.check_sulphide,
                              'func2': prP.check_sulphur,
                              'threshold': 0.5}
                             ],

    'Disulphide': [prP.check_disulphide, {}],
    'Disulphide (Aromatic)': [prP.check_pattern_aromatic,
                              {'pattern_function': prP.check_disulphide}
                              ],
    'Disulphide (Cyclic)': [prR.check_pattern_cyclic,
                            {'pattern_function': prP.check_disulphide}
                            ],
    'Disulphide-Aryl': [prP.check_pattern_aromatic_substituent,
                        {'pattern_function': prP.check_disulphide}
                        ],
    'Disulphide-Ring': [prR.check_pattern_cyclic_substituent,
                        {'pattern_function': prP.check_disulphide}
                        ],
    'Disulphide > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                               {'func1': prP.check_disulphide,
                                'func2': prP.check_sulphur,
                                'threshold': 0.5}
                               ],

    'Thiocarbamate': [prP.check_thiocarbamate, {}],
    'Thiocarbamate (Aromatic)': [prP.check_pattern_aromatic,
                                 {'pattern_function': prP.check_thiocarbamate}
                                 ],
    'Thiocarbamate (Cyclic)': [prR.check_pattern_cyclic,
                               {'pattern_function': prP.check_thiocarbamate}
                               ],
    'Thiocarbamate-Aryl': [prP.check_pattern_aromatic_substituent,
                           {'pattern_function': prP.check_thiocarbamate}
                           ],
    'Thiocarbamate-Ring': [prR.check_pattern_cyclic_substituent,
                           {'pattern_function': prP.check_thiocarbamate}
                           ],
    'Thiocarbamate > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                  {'func1': prP.check_thiocarbamate,
                                   'func2': prP.check_sulphur,
                                   'threshold': 0.5}
                                  ],

    'Thiocyanate': [prP.check_thiocyanate, {}],
    'Thiocyanate (Aromatic)': [prP.check_pattern_aromatic,
                               {'pattern_function': prP.check_thiocyanate}
                               ],
    'Thiocyanate (Cyclic)': [prR.check_pattern_cyclic,
                             {'pattern_function': prP.check_thiocyanate}
                             ],
    'Thiocyanate-Aryl': [prP.check_pattern_aromatic_substituent,
                         {'pattern_function': prP.check_thiocyanate}
                         ],
    'Thiocyanate-Ring': [prR.check_pattern_cyclic_substituent,
                         {'pattern_function': prP.check_thiocyanate}
                         ],
    'Thiocyanate > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                {'func1': prP.check_thiocyanate,
                                 'func2': prP.check_sulphur,
                                 'threshold': 0.5}
                                ],

    'Isothiocyanate': [prP.check_isothiocyanate, {}],
    'Isothiocyanate (Aromatic)': [prP.check_pattern_aromatic,
                                  {'pattern_function': prP.check_isothiocyanate}
                                  ],
    'Isothiocyanate (Cyclic)': [prR.check_pattern_cyclic,
                                {'pattern_function': prP.check_isothiocyanate}
                                ],
    'Isothiocyanate-Aryl': [prP.check_pattern_aromatic_substituent,
                            {'pattern_function': prP.check_isothiocyanate}
                            ],
    'Isothiocyanate-Ring': [prR.check_pattern_cyclic_substituent,
                            {'pattern_function': prP.check_isothiocyanate}
                            ],
    'Isothiocyanate > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_isothiocyanate,
                                    'func2': prP.check_sulphur,
                                    'threshold': 0.5}
                                   ],

    'Sulphinic acid': [prP.check_sulphinic_acid, {}],
    'Sulphinic acid-Aryl': [prP.check_pattern_aromatic_substituent,
                            {'pattern_function': prP.check_sulphinic_acid}
                            ],
    'Sulphinic acid-Ring': [prR.check_pattern_cyclic_substituent,
                            {'pattern_function': prP.check_sulphinic_acid}
                            ],
    'Sulphinic acid > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_sulphinic_acid,
                                    'func2': prP.check_sulphur,
                                    'threshold': 0.5}
                                   ],

    'Sulphinic ester': [prP.check_sulphinic_ester, {}],
    'Sulphinic ester (Aromatic)': [prP.check_pattern_aromatic,
                                   {'pattern_function': prP.check_sulphinic_ester}
                                   ],
    'Sulphinic ester (Cyclic)': [prR.check_pattern_cyclic,
                                 {'pattern_function': prP.check_sulphinic_ester}
                                 ],
    'Sulphinic ester-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_sulphinic_ester}
                             ],
    'Sulphinic ester-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_sulphinic_ester}
                             ],
    'Sulphinic ester > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_sulphinic_ester,
                                     'func2': prP.check_sulphur,
                                     'threshold': 0.5}
                                    ],

    'Sulphone': [prP.check_sulphone, {}],
    'Sulphone (Aromatic)': [prP.check_pattern_aromatic,
                            {'pattern_function': prP.check_sulphone}
                            ],
    'Sulphone (Cyclic)': [prR.check_pattern_cyclic,
                          {'pattern_function': prP.check_sulphone}
                          ],
    'Sulphone-Aryl': [prP.check_pattern_aromatic_substituent,
                      {'pattern_function': prP.check_sulphone}
                      ],
    'Sulphone-Ring': [prR.check_pattern_cyclic_substituent,
                      {'pattern_function': prP.check_sulphone}
                      ],
    'Sulphone > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                             {'func1': prP.check_sulphone,
                              'func2': prP.check_sulphur,
                              'threshold': 0.5}
                             ],

    'Carbosulphone': [prP.check_carbosulphone, {}],
    'Carbosulphone (Aromatic)': [prP.check_pattern_aromatic,
                                 {'pattern_function': prP.check_carbosulphone}
                                 ],
    'Carbosulphone (Cyclic)': [prR.check_pattern_cyclic,
                               {'pattern_function': prP.check_carbosulphone}
                               ],
    'Carbosulphone-Aryl': [prP.check_pattern_aromatic_substituent,
                           {'pattern_function': prP.check_carbosulphone}
                           ],
    'Carbosulphone-Ring': [prR.check_pattern_cyclic_substituent,
                           {'pattern_function': prP.check_carbosulphone}
                           ],
    'Carbosulphone > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                  {'func1': prP.check_carbosulphone,
                                   'func2': prP.check_sulphur,
                                   'threshold': 0.5}
                                  ],

    'Sulphonic acid': [prP.check_sulphonic_acid, {}],
    'Sulphonic acid-Aryl': [prP.check_pattern_aromatic_substituent,
                            {'pattern_function': prP.check_sulphonic_acid}
                            ],
    'Sulphonic acid-Ring': [prR.check_pattern_cyclic_substituent,
                            {'pattern_function': prP.check_sulphonic_acid}
                            ],
    'Sulphonic acid > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_sulphonic_acid,
                                    'func2': prP.check_sulphur,
                                    'threshold': 0.5}
                                   ],

    'Sulphonic ester': [prP.check_sulphonic_ester, {}],
    'Sulphonic ester (Aromatic)': [prP.check_pattern_aromatic,
                                   {'pattern_function': prP.check_sulphonic_ester}
                                   ],
    'Sulphonic ester (Cyclic)': [prR.check_pattern_cyclic,
                                 {'pattern_function': prP.check_sulphonic_ester}
                                 ],
    'Sulphonic ester-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_sulphonic_ester}
                             ],
    'Sulphonic ester-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_sulphonic_ester}
                             ],
    'Sulphonic ester > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_sulphonic_ester,
                                     'func2': prP.check_sulphur,
                                     'threshold': 0.5}
                                    ],

    'Sulphonamide': [prP.check_sulphonamide, {}],
    'Sulphonamide (Aromatic)': [prP.check_pattern_aromatic,
                                {'pattern_function': prP.check_sulphonamide}
                                ],
    'Sulphonamide (Cyclic)': [prR.check_pattern_cyclic,
                              {'pattern_function': prP.check_sulphonamide}
                              ],
    'Sulphonamide-Aryl': [prP.check_pattern_aromatic_substituent,
                          {'pattern_function': prP.check_sulphonamide}
                          ],
    'Sulphonamide-Ring': [prR.check_pattern_cyclic_substituent,
                          {'pattern_function': prP.check_sulphonamide}
                          ],
    'Sulphonamide > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                 {'func1': prP.check_sulphonamide,
                                  'func2': prP.check_sulphur,
                                  'threshold': 0.5}
                                 ],

    'Sulphoxide': [prP.check_sulphoxide, {}],
    'Sulphoxide (Aromatic)': [prP.check_pattern_aromatic,
                              {'pattern_function': prP.check_sulphoxide}
                              ],
    'Sulphoxide (Cyclic)': [prR.check_pattern_cyclic,
                            {'pattern_function': prP.check_sulphoxide}
                            ],
    'Sulphoxide-Aryl': [prP.check_pattern_aromatic_substituent,
                        {'pattern_function': prP.check_sulphoxide}
                        ],
    'Sulphoxide-Ring': [prR.check_pattern_cyclic_substituent,
                        {'pattern_function': prP.check_sulphoxide}
                        ],
    'Sulphoxide > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                               {'func1': prP.check_sulphoxide,
                                'func2': prP.check_sulphur,
                                'threshold': 0.5}
                               ],

    'Carbosulphoxide': [prP.check_carbosulphoxide, {}],
    'Carbosulphoxide (Aromatic)': [prP.check_pattern_aromatic,
                                   {'pattern_function': prP.check_carbosulphoxide}
                                   ],
    'Carbosulphoxide (Cyclic)': [prR.check_pattern_cyclic,
                                 {'pattern_function': prP.check_carbosulphoxide}
                                 ],
    'Carbosulphoxide-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_carbosulphoxide}
                             ],
    'Carbosulphoxide-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_carbosulphoxide}
                             ],
    'Carbosulphoxide > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_carbosulphoxide,
                                     'func2': prP.check_sulphur,
                                     'threshold': 0.5}
                                    ],

    'Sulphuric acid': [prP.check_sulphuric_acid, {}],
    'Sulphuric acid-Aryl': [prP.check_pattern_aromatic_substituent,
                            {'pattern_function': prP.check_sulphuric_acid}
                            ],
    'Sulphuric acid-Ring': [prR.check_pattern_cyclic_substituent,
                            {'pattern_function': prP.check_sulphuric_acid}
                            ],
    'Sulphuric acid > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_sulphuric_acid,
                                    'func2': prP.check_sulphur,
                                    'threshold': 0.5}
                                   ],

    'Sulphuric ester': [prP.check_sulphuric_ester, {}],
    'Sulphuric ester (Aromatic)': [prP.check_pattern_aromatic,
                                   {'pattern_function': prP.check_sulphuric_ester}
                                   ],
    'Sulphuric ester (Cyclic)': [prR.check_pattern_cyclic,
                                 {'pattern_function': prP.check_sulphuric_ester}
                                 ],
    'Sulphuric ester-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_sulphuric_ester}
                             ],
    'Sulphuric ester-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_sulphuric_ester}
                             ],
    'Sulphuric ester > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_sulphuric_ester,
                                     'func2': prP.check_sulphur,
                                     'threshold': 0.5}
                                    ],

    'Sulphamic acid': [prP.check_sulphamic_acid, {}],
    'Sulphamic acid-Aryl': [prP.check_pattern_aromatic_substituent,
                            {'pattern_function': prP.check_sulphamic_acid}
                            ],
    'Sulphamic acid-Ring': [prR.check_pattern_cyclic_substituent,
                            {'pattern_function': prP.check_sulphamic_acid}
                            ],
    'Sulphamic acid > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_sulphamic_acid,
                                    'func2': prP.check_sulphur,
                                    'threshold': 0.5}
                                   ],

    'Sulphamic ester': [prP.check_sulphamic_ester, {}],
    'Sulphamic ester (Aromatic)': [prP.check_pattern_aromatic,
                                   {'pattern_function': prP.check_sulphamic_ester}
                                   ],
    'Sulphamic ester (Cyclic)': [prR.check_pattern_cyclic,
                                 {'pattern_function': prP.check_sulphamic_ester}
                                 ],
    'Sulphamic ester-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_sulphamic_ester}
                             ],
    'Sulphamic ester-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_sulphamic_ester}
                             ],
    'Sulphamic ester > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_sulphamic_ester,
                                     'func2': prP.check_sulphur,
                                     'threshold': 0.5}
                                    ],

    'Sulphenic acid': [prP.check_sulphenic_acid, {}],
    'Sulphenic acid-Aryl': [prP.check_pattern_aromatic_substituent,
                            {'pattern_function': prP.check_sulphenic_acid}
                            ],
    'Sulphenic acid-Ring': [prR.check_pattern_cyclic_substituent,
                            {'pattern_function': prP.check_sulphenic_acid}
                            ],
    'Sulphenic acid > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                   {'func1': prP.check_sulphenic_acid,
                                    'func2': prP.check_sulphur,
                                    'threshold': 0.5}
                                   ],

    'Sulphenic ester': [prP.check_sulphenic_ester, {}],
    'Sulphenic ester (Aromatic)': [prP.check_pattern_aromatic,
                                   {'pattern_function': prP.check_sulphenic_ester}
                                   ],
    'Sulphenic ester (Cyclic)': [prR.check_pattern_cyclic,
                                 {'pattern_function': prP.check_sulphenic_ester}
                                 ],
    'Sulphenic ester-Aryl': [prP.check_pattern_aromatic_substituent,
                             {'pattern_function': prP.check_sulphenic_ester}
                             ],
    'Sulphenic ester-Ring': [prR.check_pattern_cyclic_substituent,
                             {'pattern_function': prP.check_sulphenic_ester}
                             ],
    'Sulphenic ester > 50% tot S': [prB.pattern_rel_fraction_greater_than,
                                    {'func1': prP.check_sulphenic_ester,
                                     'func2': prP.check_sulphur,
                                     'threshold': 0.5}
                                    ],

    # Halogens

    'Fluorine': [prP.check_fluorine, {}],
    'Fluorine > 50% tot X': [prB.pattern_rel_fraction_greater_than,
                             {'func1': prP.check_fluorine,
                              'func2': prP.check_halogen,
                              'threshold': 0.5}
                             ],
    'Chlorine': [prP.check_chlorine, {}],
    'Chlorine > 50% tot X': [prB.pattern_rel_fraction_greater_than,
                             {'func1': prP.check_chlorine,
                              'func2': prP.check_halogen,
                              'threshold': 0.5}
                             ],
    'Bromine': [prP.check_bromine, {}],
    'Bromine > 50% tot X': [prB.pattern_rel_fraction_greater_than,
                            {'func1': prP.check_bromine,
                             'func2': prP.check_halogen,
                             'threshold': 0.5}
                            ],
    'Iodine': [prP.check_iodine, {}],
    'Iodine > 50% tot X': [prB.pattern_rel_fraction_greater_than,
                           {'func1': prP.check_iodine,
                            'func2': prP.check_halogen,
                            'threshold': 0.5}
                           ],
    'Halogen': [prP.check_halogen, {}],
    'Halogen-Aryl': [prP.check_pattern_aromatic_substituent,
                     {'pattern_function': prP.check_halogen}
                     ],
    'Halogen-Ring': [prR.check_pattern_cyclic_substituent,
                     {'pattern_function': prP.check_halogen}
                     ],
    'Haloalkane': [prP.check_haloalkane, {}],
    'Haloalkane (Primary)': [prP.check_haloalkane_primary, {}],
    'Haloalkane (Secondary)': [prP.check_haloalkane_secondary, {}],
    'Haloalkane (Tertiary)': [prP.check_haloalkane_tertiary, {}],
    'Haloalkene': [prP.check_haloalkane, {}],
    'Halogen-Carbon': [prP.check_halogen_carbon, {}],
    'Halogen-Nitrogen': [prP.check_halogen_nitrogen, {}],
    'Halogen-Oxygen': [prP.check_halogen_oxygen, {}],
    'Oxohalide': [prP.check_oxohalide, {}],

    # other groups

    'Alkali Metals (G1)': [prP.check_alkali_metals,
                           {}],
    'Alkaline Earth Metals (G2)': [prP.check_alkaline_earth_metals,
                                   {}],
    'Transition Metals (G3-12)': [prP.check_transition_metals,
                                  {}],
    'Boron Group Elements (G13)': [prP.check_boron_group_elements,
                                   {}],
    'Carbon Group Elements, no C (G14)': [prP.check_carbon_group_elements,
                                          {}],
    'Nitrogen Group Elements, no N/P (G15)': [prP.check_nitrogen_group_elements,
                                              {}],
    'Chalcogens, no O/S (G16)': [prP.check_chalcogens, {}],
    'Noble Gases (G18)': [prP.check_noble_gases, {}],

    # other structures #

    'Charge +': [prP.check_pos_charge_1, {}],
    '2 Charge +': [prP.check_pos_charge_2, {}],
    '>3 Charge + ': [prP.check_pos_charge_3, {}],
    'Charge -': [prP.check_pos_charge_1, {}],
    '2 Charge -': [prP.check_pos_charge_2, {}],
    '>3 Charge - ': [prP.check_pos_charge_3, {}],

    'Unbranched structure >=5': [prP.check_unbranched_structure,
                                 {'n_units': 5}
                                 ],
    'Unbranched structure >=10': [prP.check_unbranched_structure,
                                  {'n_units': 10}
                                  ],
    'Unbranched structure >=15': [prP.check_unbranched_structure,
                                  {'n_units': 15}
                                  ],
    'Unbranched structure >=20': [prP.check_unbranched_structure,
                                  {'n_units': 20}
                                  ],

    'Unbranched rotatable chain >=5': [prP.check_unbranched_rotatable_chain,
                                       {'n_units': 5}
                                       ],
    'Unbranched rotatable chain >=10': [prP.check_unbranched_rotatable_chain,
                                        {'n_units': 10}
                                        ],
    'Unbranched rotatable chain >=15': [prP.check_unbranched_rotatable_chain,
                                        {'n_units': 15}
                                        ],
    'Unbranched rotatable chain >=20': [prP.check_unbranched_rotatable_chain,
                                        {'n_units': 20}
                                        ],

    'Unbranched rotatable carbons >=5': [prP.check_unbranched_rotatable_carbons,
                                         {'n_units': 5}
                                         ],
    'Unbranched rotatable carbons >=10': [prP.check_unbranched_rotatable_carbons,
                                          {'n_units': 10}
                                          ],
    'Unbranched rotatable carbons >=15': [prP.check_unbranched_rotatable_carbons,
                                          {'n_units': 15}
                                          ],
    'Unbranched rotatable carbons >=20': [prP.check_unbranched_rotatable_carbons,
                                          {'n_units': 20}
                                          ],

    'Rotatable bonds': [prBn.check_rotatable_bonds, {}],
    'Rotatable bonds > 20% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                        {'func1': prBn.check_rotatable_bonds,
                                         'func2': prBn.check_bonds,
                                         'threshold': 0.2}
                                        ],
    'Rotatable bonds > 50% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                        {'func1': prBn.check_rotatable_bonds,
                                         'func2': prBn.check_bonds,
                                         'threshold': 0.5}
                                        ],
    'Rotatable bonds > 80% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                        {'func1': prBn.check_rotatable_bonds,
                                         'func2': prBn.check_bonds,
                                         'threshold': 0.8}
                                        ],

    'Cyclic bonds': [prBn.check_cyclic_bonds, {}],
    'Cyclic bonds > 20% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prBn.check_cyclic_bonds,
                                      'func2': prBn.check_bonds,
                                      'threshold': 0.2}
                                     ],
    'Cyclic bonds > 50% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prBn.check_cyclic_bonds,
                                      'func2': prBn.check_bonds,
                                      'threshold': 0.5}
                                     ],

    'Aromatic bonds': [prBn.check_aromatic_bonds, {}],
    'Aromatic bonds > 50% tot cyclic bonds': [prB.pattern_rel_fraction_greater_than,
                                              {'func1': prBn.check_aromatic_bonds,
                                               'func2': prBn.check_cyclic_bonds,
                                               'threshold': 0.5}
                                              ],
    'Aromatic bonds > 20% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                       {'func1': prBn.check_aromatic_bonds,
                                        'func2': prBn.check_bonds,
                                        'threshold': 0.2}
                                       ],

    'Single bonds': [prBn.check_single_bonds, {}],
    'Single bonds > 20% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prBn.check_single_bonds,
                                      'func2': prBn.check_bonds,
                                      'threshold': 0.2}
                                     ],
    'Single bonds > 50% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prBn.check_single_bonds,
                                      'func2': prBn.check_bonds,
                                      'threshold': 0.5}
                                     ],

    'Double bonds': [prBn.check_double_bonds, {}],
    'Double bonds > 20% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prBn.check_double_bonds,
                                      'func2': prBn.check_bonds,
                                      'threshold': 0.2}
                                     ],

    'Triple bonds': [prBn.check_triple_bonds, {}],
    'Triple bonds > 10% tot bonds': [prB.pattern_rel_fraction_greater_than,
                                     {'func1': prBn.check_triple_bonds,
                                      'func2': prBn.check_bonds,
                                      'threshold': 0.1}
                                     ],

    'Ring 3': [prR.check_ring_size, {'size': 3}],
    'Ring 4': [prR.check_ring_size, {'size': 4}],
    'Ring 5': [prR.check_ring_size, {'size': 5}],
    'Ring 6': [prR.check_ring_size, {'size': 6}],
    'Ring > 7': [prR.check_macrocycle, {}],

    'Ortho-substituted benzene': [prR.check_ortho_substituted_aromatic_r6, {}],
    'Meta-substituted benzene': [prR.check_meta_substituted_aromatic_r6, {}],
    'Para-substituted benzene': [prR.check_para_substituted_aromatic_r6, {}],

    'Fused rings': [prR.check_ring_fusion, {}],
    'Fused rings (Aromatic)': [prP.check_pattern_aromatic,
                               {'pattern_function': prR.check_ring_fusion}
                               ],

    'Heterocycle': [prR.check_heterocycle, {}],
    'Heterocycle (Aromatic)': [prP.check_pattern_aromatic,
                               {'pattern_function': prR.check_heterocycle}
                               ],
    'N-Heterocycle': [prR.check_heterocycle_N, {}],
    'N-Heterocycle (Aromatic)': [prP.check_pattern_aromatic,
                                 {'pattern_function': prR.check_heterocycle_N}
                                 ],
    'O-Heterocycle': [prR.check_heterocycle_O, {}],
    'O-Heterocycle (Aromatic)': [prP.check_pattern_aromatic,
                                 {'pattern_function': prR.check_heterocycle_O}
                                 ],
    'S-Heterocycle': [prR.check_heterocycle_S, {}],
    'S-Heterocycle (Aromatic)': [prP.check_pattern_aromatic,
                                 {'pattern_function': prR.check_heterocycle_S}],

    'Zwitterion': [prP.check_zwitterion, {}],
    'H-Bond acceptors': [prP.check_hbond_acceptors, {}],
    'H-Bond acceptors > 2': [prP.check_hbond_acceptors_higher_than,
                             {'n': 2}
                             ],
    'H-Bond acceptors > 5': [prP.check_hbond_acceptors_higher_than,
                             {'n': 5}
                             ],
    'H-Bond donors': [prP.check_hbond_donors, {}],
    'H-Bond donors > 2': [prP.check_hbond_donors_higher_than,
                          {'n': 2}
                          ],
    'H-Bond donors > 5': [prP.check_hbond_donors_higher_than,
                          {'n': 5}
                          ],
    }

bokeh_dictionary = {'title_location': 'above',
                    'title_fontsize': '25px',
                    'title_align': 'center',
                    'title_background_fill_colour': 'white',
                    'title_text_colour': 'black',
                    'legend_location': 'top_left',
                    'legend_title': 'Class',
                    'legend_label_text_font': 'times',
                    'legend_label_text_font_style': 'italic',
                    'legend_label_text_colour': 'navy',
                    'legend_border_line_width': 3,
                    'legend_border_line_colour': 'navy',
                    'legend_border_line_alpha': 0.5,
                    'legend_background_fill_colour': 'grey',
                    'legend_background_fill_alpha': 0.2,
                    'xaxis_label': 'DIM_1',
                    'xaxis_line_width': 3,
                    'xaxis_line_colour': 'grey',
                    'xaxis_major_label_text_colour': 'navy',
                    'xaxis_major_label_orientation': 'horizontal',
                    'yaxis_label': 'DIM_2',
                    'yaxis_line_width': 3,
                    'yaxis_line_colour': 'grey',
                    'yaxis_major_label_text_colour': 'navy',
                    'yaxis_major_label_orientation': 'vertical',
                    'axis_minor_tick_in': -3,
                    'axis_minor_tick_out': 6,
                    }

bokeh_tooltips = """
                <div>
                    <div>
                        <img
                            src="@MOLFILE" height="100" alt="@molfile" width="100"
                            style="float: top; margin: 10px 10px 10px 10px;"
                            border="0"
                        ></img>
                    </div>
                    <div>
                        <b><span style="font-size: 10px;">Name:</b> @NAME_SHORT</span>
                    </div>
                    <div>
                        <b><span style="font-size: 10px;">Class:</b> @CLASS</span>
                    </div>
                    <div>
                        <b><span style="font-size: 10px;">Mol #:</b> @index</span>
                    </div>
                    <div>
                        <b><span style="font-size: 10px;">Metadata:</b> @METADATA</span>
                    </div>
                </div>
                """

interpretable_descriptors_rdkit = [
                            'MolWt', 'HeavyAtomMolWt', 'NumValenceElectrons',
                            'NumRadicalElectrons', 'HeavyAtomCount',
                            'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
                            'NumAliphaticHeterocycles', 'NumAliphaticRings',
                            'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
                            'NumAromaticRings', 'NumHAcceptors', 'NumHDonors',
                            'NumHeteroatoms', 'NumRotatableBonds',
                            'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles',
                            'NumSaturatedRings', 'RingCount', 'MolLogP',
                             ]

interpretable_descriptors_mordred = ['nAcid', 'nBase', 'nAromAtom', 'nAromBond',
                                     'nAtom', 'nHeavyAtom', 'nSpiro',
                                     'nBridgehead', 'nHetero', 'nH', 'nB',
                                     'nC', 'nN', 'nO', 'nS', 'nP', 'nF', 'nCl',
                                     'nBr', 'nI', 'nX', 'nBondsO', 'nBondsS',
                                     'nBondsD', 'nBondsT', 'nBondsA',
                                     'nBondsM', 'nBondsKS', 'nBondsKD', 'TASA',
                                     'TPSA', 'C1SP1', 'C2SP1', 'C1SP2',
                                     'C2SP2', 'C3SP2', 'C1SP3', 'C2SP3',
                                     'C3SP3', 'C4SP3', 'HybRatio', 'FCSP3',
                                     'nRing', 'n3Ring', 'n4Ring', 'n5Ring',
                                     'n6Ring', 'n7Ring', 'n8Ring', 'n9Ring',
                                     'n10Ring', 'n11Ring', 'n12Ring',
                                     'nG12Ring', 'nHRing', 'n3HRing',
                                     'n4HRing', 'n5HRing', 'n6HRing',
                                     'n7HRing', 'n8HRing', 'n9HRing',
                                     'n10HRing', 'n11HRing', 'n12HRing',
                                     'nG12HRing', 'naRing', 'n3aRing',
                                     'n4aRing', 'n5aRing', 'n6aRing',
                                     'n7aRing', 'n8aRing', 'n9aRing',
                                     'n10aRing', 'n11aRing', 'n12aRing',
                                     'nG12aRing', 'naHRing', 'n3aHRing',
                                     'n4aHRing', 'n5aHRing', 'n6aHRing',
                                     'n7aHRing', 'n8aHRing', 'n9aHRing',
                                     'n10aHRing', 'n11aHRing', 'n12aHRing',
                                     'nG12aHRing', 'nARing', 'n3ARing',
                                     'n4ARing', 'n5ARing', 'n6ARing',
                                     'n7ARing', 'n8ARing', 'n9ARing',
                                     'n10ARing', 'n11ARing', 'n12ARing',
                                     'nG12ARing', 'nAHRing', 'n3AHRing',
                                     'n4AHRing', 'n5AHRing', 'n6AHRing',
                                     'n7AHRing', 'n8AHRing', 'n9AHRing',
                                     'n10AHRing', 'n11AHRing', 'n12AHRing',
                                     'nG12AHRing', 'nFRing', 'n3FRing',
                                     'n4FRing', 'n5FRing', 'n6FRing',
                                     'n7FRing', 'n8FRing', 'n9FRing',
                                     'n10FRing', 'n11FRing', 'n12FRing',
                                     'nG12FRing', 'nFHRing', 'n3FHRing',
                                     'n4FHRing', 'n5FHRing', 'n6FHRing',
                                     'n7FHRing', 'n8FHRing', 'n9FHRing',
                                     'n10FHRing', 'n11FHRing', 'n12FHRing',
                                     'nG12FHRing', 'nFaRing', 'n3FaRing',
                                     'n4FaRing', 'n5FaRing', 'n6FaRing',
                                     'n7FaRing', 'n8FaRing', 'n9FaRing',
                                     'n10FaRing', 'n11FaRing', 'n12FaRing',
                                     'nG12FaRing', 'nFaHRing', 'n3FaHRing',
                                     'n4FaHRing', 'n5FaHRing', 'n6FaHRing',
                                     'n7FaHRing', 'n8FaHRing', 'n9FaHRing',
                                     'n10FaHRing', 'n11FaHRing', 'n11FaHRing',
                                     'n12FaHRing', 'nFARing', 'n3FARing',
                                     'n4FARing', 'n5FARing', 'n6FARing',
                                     'n7FARing', 'n8FARing', 'n9FARing',
                                     'n10FARing', 'n11FARing', 'n12FARing',
                                     'nG12FARing', 'nFAHRing', 'n3FAHRing',
                                     'n4FAHRing', 'n5FAHRing', 'n6FAHRing',
                                     'n7FAHRing', 'n8FAHRing', 'n9FAHRing',
                                     'n10FAHRing', 'n11FAHRing', 'n12FAHRing',
                                     'nG12FAHRing', 'nRot', 'RotRatio', 'SLogP'
                                     ]

similarity_metric_dictionary = {'Tanimoto': metrics.
                                TanimotoSimilarity,
                                'Dice': metrics.
                                DiceSimilarity,
                                'Cosine': metrics.
                                CosineSimilarity,
                                'Sokal': metrics.
                                SokalSimilarity,
                                'Russel': metrics.
                                RusselSimilarity,
                                'RogotGoldberg': metrics.
                                RogotGoldbergSimilarity,
                                'AllBit': metrics.
                                AllBitSimilarity,
                                'OnBit': metrics.
                                OnBitSimilarity,
                                'Kulczynski': metrics.
                                KulczynskiSimilarity,
                                'McConnaughey': metrics.
                                McConnaugheySimilarity,
                                'Asymmetric': metrics.
                                AsymmetricSimilarity,
                                'BraunBlanquet': metrics.
                                BraunBlanquetSimilarity,
                                'Tversky': metrics.
                                TverskySimilarity,
                                }
