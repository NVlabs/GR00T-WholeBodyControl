import inspect
import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
from lxml import etree as ET
from robosuite.utils.mjcf_utils import find_elements

import robocasa

TEXTURES_DIR = Path(inspect.getfile(robocasa)).parent / "models" / "assets" / "generative_textures"

CABINET_TEX_NAMES = [
    "flat copy 26.png",
    "196415a6-3c0f-4a1f-a644-fc54a6801223.png",
    "2a310e7f-8005-4dba-8dea-edb1484e1e28.png",
    "flat copy 37.png",
    "1fb8c016-b655-48ab-ad53-d6238a745065.png",
    "f80067c5-9967-4607-880f-1c4002d400cb.png",
    "00846c84-f900-4bdd-897f-3928e7d87dce.png",
    "flat copy 38.png",
    "54c4d59e-c0b1-4608-a953-568e9ae13409.png",
    "flat copy 2.png",
    "flat copy 23.png",
    "flat copy 16.png",
    "251695ed-d413-48fe-817e-c3e02b3ee117.png",
    "18338c06-57fc-4882-ad44-b5f3b3846806.png",
    "flat copy 40.png",
    "36a1e6cb-98a1-4ada-a14e-8c04f135c5ee.png",
    "f06bca10-a7ce-4a4a-92f6-973d7400fe8a.png",
    "flat copy 41.png",
    "397c17e0-e91b-4fa6-98d9-97e6d2f009ed.png",
    "532a9660-8c7a-488b-8930-9134a9479afa.png",
    "flat copy 33.png",
    "d0299fe2-026b-4727-85e1-cb5de303e315.png",
    "68a933bc-df0c-4891-bd41-2c05e4fde504.png",
    "6698bb80-52f7-4e5b-ad6a-4a8122504f9a.png",
    "99f0113d-8cb4-40b6-9670-b4be4c1fc475.png",
    "flat copy 32.png",
    "flat copy 34.png",
    "c1ad279c-9093-469e-aa20-b73af0fc8db4.png",
    "518ddc39-55c2-4b18-8bf6-e9c4f0806760.png",
    "flat copy 8.png",
    "860cab10-089c-4d99-ae89-3e4f87df5d9c.png",
    "f29d41e9-d285-47e4-ad29-f04423533802.png",
    "flat copy 13.png",
    "flat copy 12.png",
    "e6420ee4-14f0-404f-9cd7-abc9a69da621.png",
    "flat copy 4.png",
    "de7f86d5-56c9-4331-a9aa-655687dc354b.png",
    "flat copy 35.png",
    "flat.png",
    "b6abcb20-8b90-42d3-9175-ba6b4d53a7a9.png",
    "d29a6530-28d6-4dbe-bce5-9014d613ae5e.png",
    "992f49b2-3f7c-4c54-a4d6-75af4693353b.png",
    "flat copy 27.png",
    "fd195b15-b98c-47c9-b608-de4157f6ceba.png",
    "flat copy 10.png",
    "flat copy 15.png",
    "af993127-4ec6-4fa3-9a27-018af985a4a0.png",
    "8de5fd51-b7b5-4014-a892-f0498c35fec2.png",
    "flat copy 3.png",
    "flat copy 11.png",
    "flat copy 30.png",
    "edc5b189-60a3-4aa0-a68d-cc35009f3648.png",
    "flat copy 20.png",
    "flat copy 19.png",
    "9fe8882d-3aa9-4d2b-8de3-d9d220ea526c.png",
    "flat copy 7.png",
    "flat copy 46.png",
    "flat copy 9.png",
    "230e3f37-02d7-4be9-b6a8-1e4fb3acd368.png",
    "flat copy 45.png",
    "7f426b18-12ee-492b-9c69-dfa8ccbc42cc.png",
    "flat copy 49.png",
    "flat copy 22.png",
    "7fe6e2dc-615f-432e-92d4-0a4f17a9ac49.png",
    "flat copy 47.png",
    "flat copy 42.png",
    "34c9cf7b-5b5d-4c81-bb95-6d5743d25a18.png",
    "aae9ec57-7df1-4af7-8e40-62962b90cce4.png",
    "flat copy 36.png",
    "flat copy 25.png",
    "5a34b72d-3b7e-4a9a-9a0f-f73f2230fe28.png",
    "e2e0160d-9809-48d2-b018-416ee0254fc0.png",
    "a6bb714e-0f35-46d6-b7e8-3c18232a4801.png",
    "a86a130e-04a3-44b0-99b7-44ce0a63cc2b.png",
    "01f783e0-38de-497d-8ebe-c8d8a47e520a.png",
    "flat copy 44.png",
    "e0e2d63e-900b-4ecb-8df9-28115a58c3a2.png",
    "flat copy 50.png",
    "flat copy 5.png",
    "flat copy 18.png",
    "1aac54cf-8bf3-428c-93cd-1d3c258ff6b4.png",
    "flat copy 14.png",
    "11717454-440d-4b32-85c3-bf0013c66c61.png",
    "67d50012-cba6-4525-8e00-b71124861f7e.png",
    "25436e21-3140-4875-89a9-d14c5ce20f31.png",
    "e4a28073-2252-4603-8c4b-75527a8c5ae9.png",
    "flat copy 17.png",
    "flat copy 48.png",
    "flat copy 6.png",
    "28d31de4-53f8-434a-ac57-4d5d27f5ab2f.png",
    "flat copy.png",
    "flat copy 39.png",
    "flat copy 43.png",
    "flat copy 24.png",
    "6a13b80d-83c7-4fe1-8412-4fc07eefaa35.png",
    "43a60b06-97b9-4119-b416-c7fc6783aa81.png",
    "flat copy 21.png",
    "flat copy 28.png",
    "flat copy 31.png",
    "flat copy 29.png",
]

COUNTER_TOP_TEX_NAMES = [
    "5065776e-1fbc-4a70-a786-9a2ec6ee5aa0.png",
    "e7f151aa-2354-48d0-94e8-071462f16ca7.png",
    "b6618923-8455-44c3-8ae6-b766f34267cc.png",
    "71d72043-3d97-4ea6-82f0-3ccbb53e0534.png",
    "deb1e565-5c29-4285-aef8-f904c2b41dfb.png",
    "ecf23b20-3db2-476a-bc06-442754ba5e76.png",
    "39be4639-f28b-4bfa-8c1e-4b37e40395be.png",
    "8bafdd4c-a44c-4bc0-8978-e7bc0abcedb4.png",
    "3b958a28-0807-484a-92e7-f17b4c6dcbeb.png",
    "ecd3c3fb-c4f2-4021-86d8-665e92b2735e.png",
    "014ac251-0435-41c2-9dee-c664ee268327.png",
    "0171bbba-6293-4117-b0ad-bc4b98d1c20e.png",
    "0bd64a61-e098-4d06-ba3f-0a12269d3584.png",
    "d5db5c5c-779b-4de0-ac66-31856239d488.png",
    "9c47a3ac-2e56-47c1-86d4-006f98e21987.png",
    "133b4eaf-bb93-4655-844c-a02278dfc032.png",
    "58bcfa8c-6e11-494f-8fc5-d5a32ac7b3de.png",
    "276de88a-374b-4dde-ac45-c52d651044b3.png",
    "f52f1aaa-c061-42c0-8bdd-d2edb4af8736.png",
    "1ccd1303-2e23-44c4-bb21-0dd474b0d899.png",
    "761d55a2-5858-4d02-ab13-0bb272a67eed.png",
    "c9d714cd-b00b-4d96-b6a6-cdc9e01c1b71.png",
    "06da7e0e-c12c-4da7-a56b-9a3c82d87809.png",
    "e5d1c208-34ad-49a7-8151-c0d55ecff952.png",
    "9ccbd07a-887c-48c8-a806-0fac93606bb0.png",
    "3d48ef42-02d9-41ed-bef6-f5c0dd47cdfa.png",
    "9e1fd142-1d1c-4a82-a961-681db31b343f.png",
    "5259cbe7-d956-4a0a-99d1-d7c7d391488f.png",
    "94bdb3c5-faa2-4274-9337-ca5b9fabce88.png",
    "e572f3d8-9ef2-4cd2-83c1-8acb766dbc4b.png",
    "956782b3-ca4b-49e8-8939-6c50e87b9609.png",
    "258270ae-3af4-4644-a031-176efac2c3f9.png",
    "1fa52c2f-7474-4aa9-9ff9-0e411a78daac.png",
    "f1299a54-59e5-42ea-9f53-e9f11fd6ca3c.png",
    "6a6a86a2-bbba-422a-9620-5b90adaaf7a3.png",
    "771c03d6-f0c8-4e3e-b734-0a07fff03c22.png",
    "8c6ed656-6d1c-4707-82a2-cbff87a589f7.png",
    "6a4f994f-293f-4a8a-8f8d-ac99f8e58adf.png",
    "da90fb4d-3dfb-4a17-8ecc-95e82ab69b32.png",
    "1563c9ee-30ab-44cc-b21f-fab5ad8df99d.png",
    "932f5612-9058-4923-820d-894d630190cb.png",
    "5e2958ab-c27d-41c8-8167-de9053f79038.png",
    "2408cfed-ad91-43e2-9295-fdafafb76c8d.png",
    "8549466d-50fe-4cb9-9001-e93532e7938c.png",
    "00e4b598-2dd0-4ff9-b0a4-46e223a9296e.png",
    "0666bf9a-4625-4132-bd7f-0079dace8c8c.png",
    "ceb18b7d-1dcf-4aa2-adae-0e0aae1c5eeb.png",
    "a4903060-7bfd-4bd5-b8d6-b916d172a82b.png",
    "f5f4cf81-db71-42c7-b889-c25c145132c6.png",
    "bf277536-38a6-41cd-a278-bd68e25816fd.png",
    "a1cfe2ab-12cd-4d43-9ffc-77d48d0493f3.png",
    "1c72f7db-374f-47db-8424-9526d4e7fa7a.png",
    "3dc423ca-b351-429f-9842-1bd17e3ce360.png",
    "2a8aea9d-218c-4601-8e8d-b98eb93fa461.png",
    "451b6b8b-5d86-44ad-853b-e99153ba48e1.png",
    "25d4c91a-c831-4249-b78e-a4b25d0b1ea1.png",
    "b4f0036f-f2e7-4167-bbc4-708ee783052f.png",
    "db04923e-96ac-41db-a16c-901f10a41bf0.png",
    "65e832aa-974a-4ca8-9dd8-e90a96d75b6c.png",
    "f66bf802-8273-4df7-a58a-faa8ba6ed46f.png",
    "827976d2-dbab-47a3-905f-602e945a6b30.png",
    "124bd263-2b8a-4057-9211-b36869047fd2.png",
    "f69e2027-7163-4a53-8361-6050e5679242.png",
    "d366b33d-335e-4a76-91f5-a5ce14266c89.png",
    "adcb6ab3-7ecf-47f9-a11e-6a505ad23d98.png",
    "2cea6a36-7053-4808-be22-fc5cc76b5531.png",
    "ac3491a2-e817-4f5e-8453-ce34cb78a8a9.png",
    "a875270b-fd82-409d-9bed-1d567657ff8e.png",
    "f29cfb8b-d640-48a4-ae5e-02395ed7c6db.png",
    "8c165058-abec-4ea9-9990-19b7eff4df4f.png",
    "1c6ea15c-1cad-4a0c-838a-5eee8f99cfc9.png",
    "897d7c04-f1bd-4bb1-bffb-bfada23a5b58.png",
    "1b23c19f-7e79-4b96-a744-42615a72d722.png",
    "f921b606-5363-41c2-aa93-dc8ec1475fda.png",
    "efca4079-dd9f-4d54-8943-216d6c02e70a.png",
    "040ff758-026f-4313-b390-ec86d3690f9d.png",
    "0104c474-6473-4e05-b789-6bdb59f55da7.png",
    "f73ed4a7-c1b1-4660-9eba-00f2ba25977b.png",
    "0b184f66-ba4f-4fcd-9556-bea9f58dc9c6.png",
    "7f638e9b-10e8-448c-a2f9-a75ae414dcfa.png",
    "e563fd4b-656f-45f1-b803-df0d6dcb8bd3.png",
    "0ed8ccf8-33e5-45a2-9200-4dc1a5565494.png",
    "1ab263d2-ea24-4fe7-9383-f227aadc8c47.png",
    "561d2d94-ec96-4246-974a-d9abc6f2297a.png",
    "9af6ae6e-a842-4baa-a01e-cec57045c367.png",
    "1c6fc0f3-d4ef-423b-9bba-d4a5d59c05df.png",
    "76a72b39-24ec-4044-aad8-49f381939063.png",
    "08ca0b3f-17e9-476e-9c17-ff81616c6bd7.png",
    "9b637968-8173-44da-8f41-3943c4d182f9.png",
    "725eb22a-2c4b-44b0-8aaf-bef282232601.png",
    "3e81b87f-10ee-4f74-8ba8-1b724df383fb.png",
    "fc9c7e5a-7abb-4ff6-84f2-9fce43bc200f.png",
    "a8f33495-cd01-4b66-9c7e-3ba5edec6565.png",
    "8585099d-cf7f-4c1c-8f36-937939dba42e.png",
    "fe8afe85-5153-48cb-80e0-88de2a577f0c.png",
    "46c59505-7dec-48f2-a0d1-c36ac213890e.png",
    "95b636b0-8afb-4430-8f6a-1f4e0f4eb808.png",
    "5ab4d96a-5401-45d8-977c-b3fef4e55ca9.png",
    "8c8dd1cd-6c84-4fa8-b51b-ac736fda7a6c.png",
    "76c7b19f-b375-48db-a1df-d79485a86d94.png",
]

FLOOR_TEX_NAMES = [
    "8e4ea598-7170-4d24-ad12-42d45ef17a3b.png",
    "9d7e7eac-9d81-4b04-8a07-ec35f9e7b58d.png",
    "83c7c8bf-9277-4f59-88bd-1a4f428262ef.png",
    "28359684-bb12-4b99-86b2-8c0998ee8fec.png",
    "8ebc0bbe-1656-465e-b31e-766d1aceaec2.png",
    "ada4e736-880d-4372-b148-218dc114c7fe.png",
    "8a12dcf7-fc25-4341-9ea0-e583f1cc3d71.png",
    "28f34ad5-b4f1-4b81-99cc-b04ed28d653b.png",
    "b5350f4e-9521-4af1-a03f-a39e5d8b1796.png",
    "14b4a5c3-69e0-425c-b8c5-ce495ab87409.png",
    "022ec5c1-324e-4217-9f03-92b0e2f9c74f.png",
    "9f345d0f-a768-4def-bebf-b818cf0c68dc.png",
    "49de13f1-9488-4893-8398-48b0a5f27453.png",
    "a938167f-6cb2-4e87-94f5-185eac06ff21.png",
    "2aa2a926-fe72-4221-bd30-16e0a0cc9324.png",
    "1671701f-60d2-4c71-95cc-8bdbb9f57ff9.png",
    "8b24233c-7658-4d52-809d-a6fc71f37554.png",
    "9582e8cc-89f5-464f-b183-2f21b8f0f267.png",
    "8184ce85-7db2-4b23-908b-1170cc6830cf.png",
    "32a18f41-8e7c-4a9d-b5ac-a7ba19a9ed35.png",
    "56c9fb44-a8ee-469d-85d4-6d419f9dcba3.png",
    "7ff039e9-7563-4ef0-b2ec-ce3d0d259a99.png",
    "dbdd0778-8700-4967-bb52-75bbfa7eb29c.png",
    "50578da0-60d8-4e12-800e-3df2fbf1bd5a.png",
    "cedd99bd-8f8f-4d60-98e8-75e0af9b7e4f.png",
    "b6c12637-3bd8-41d4-a787-3c2c271f4221.png",
    "22c6be8e-a027-47dc-b2a1-3284cf5baad0.png",
    "ed301cb6-fc33-4a6a-a2fd-4ce27a1f5bac.png",
    "fc6bffaf-b9fd-4d51-a795-cb95d280dcab.png",
    "310661bb-e74a-4b55-b999-ae1e2855990f.png",
    "c327867a-02ea-4629-8059-181d3f341dc2.png",
    "67a6f678-5f4b-4999-88fd-d0587299eb90.png",
    "c65c9338-2903-4c6a-8015-dc606d24e659.png",
    "e584ce4a-ff4d-461f-989e-5e4d21da42de.png",
    "8776902b-74c1-4a54-94f2-320b4e89b5df.png",
    "b588fa60-819c-4ae7-84bb-8f3de009ee1b.png",
    "bdddfbd0-acc5-4e73-becb-c91f69986201.png",
    "1f13ee23-d3f8-4fe4-b24d-97c4dece35c5.png",
    "447fc81b-73c3-4502-8de7-9db654e62902.png",
    "f78f9be1-385c-43ed-a89b-a4f11d45695f.png",
    "c82d1a06-f78d-4e79-9b3e-23c4756d555d.png",
    "573d34d2-a045-455c-b525-34a8571808eb.png",
    "07f74135-07ab-4e29-bb00-4782f15ab59f.png",
    "bec253e6-b2f0-48f2-afba-9207388a7937.png",
    "0139f959-a4a8-4504-b6d2-f9351afd5941.png",
    "8373a547-a792-438e-b941-f01ca1f0fb5d.png",
    "b37ff2ac-f755-42f1-bdd4-a3a0f30fbfc6.png",
    "f068688f-4b20-4d2f-b33a-9732b5050f72.png",
    "c707d14d-3089-4a3e-a5b2-1088eda621bc.png",
    "81ea9eea-29ea-4516-9e0d-e7ed0df98f5d.png",
    "9aa36664-4ac6-4a50-b0e0-05c96f52c58e.png",
    "c85db92d-b5a4-4fc8-aabd-9d4c82f3f444.png",
    "df431f5c-699e-47ff-9161-b6e264a82163.png",
    "4d8ea3e7-4a0b-4015-802c-cbbd0b12ba2e.png",
    "3a34566d-b31f-4d3d-a86a-8038fde33b8c.png",
    "83c6fd2f-caa2-4d53-b4d2-e19a7f12e6b4.png",
    "ef044eea-9ebe-475e-bc05-d4e9d9255e39.png",
    "02b897ed-2381-403a-aef1-095096993039.png",
    "bb9ea1a0-7bfb-4aa2-85ee-0dbdb5001fa3.png",
    "2afa8673-57cc-4953-b178-b57eedd9ad0a.png",
    "2066e385-7798-47b3-89cc-1eaad15c75c4.png",
    "45639cb9-7ac7-4d8c-a8a1-719d3a53d531.png",
    "8f7fabcb-6875-4437-9d9c-27a2552c10dd.png",
    "ce8ccc5f-20b9-4d06-aee6-fe85a6c02eaa.png",
    "11fa953e-c11d-4dad-8e51-47cc6dc67aeb.png",
    "8f51e06a-7bd0-4d2f-9a20-7821a975559a.png",
    "1ef4cfa4-cf52-4efd-9fb3-6698650ee239.png",
    "4d974231-cd17-4da5-ad9a-7faf83c2d683.png",
    "cadaada3-e8f1-4bc7-af71-342215edbbb1.png",
    "fb9b36a4-d87a-4c7f-a06c-a1e94869b96b.png",
    "5daabcbd-0ce4-4dba-9342-c74d3da4fadf.png",
    "612b00d2-819f-46ac-a599-38672184c842.png",
    "0b5b5a9a-1435-4dfe-b6f4-19f4dbae1e8e.png",
    "11f6a3bb-c36c-45c2-a429-501ac0cabc02.png",
    "cf96c697-41f0-4d3c-a1d4-17d0d82719d0.png",
    "81d9e9f2-8ff4-4d45-b1dd-d3afd3c62fb0.png",
    "82036c07-5321-47c8-974c-27d58e526ba1.png",
    "e793023f-7e5f-4211-9724-9022d3c8f920.png",
    "0c01c256-4779-4093-a0b9-3347c31ef53f.png",
    "9db099e2-56cb-4bec-becc-eb69c298866e.png",
    "05f75518-7b70-4098-8983-aca459f33ddb.png",
    "7c935010-badd-4fa4-8089-4fc9728dfe96.png",
    "f9d057b1-5843-4b68-a58b-9fe87291a72f.png",
    "9354be03-5a87-4ce8-85bc-8b0157abeb1c.png",
    "9bfe963b-6d17-4704-8a9f-c2de48491b70.png",
    "3b70d093-e264-4550-b5e5-f750ba577b78.png",
    "67837bbb-05ae-4665-90e5-572130b7fc86.png",
    "d339c422-36d0-40f3-8393-24076abb279c.png",
    "5ccc395f-efd5-44af-b5f9-885789e26661.png",
    "6d58d55a-34cc-455f-8b4c-004fabef6e11.png",
    "5729f1d0-0796-4d82-99f7-76a171266fae.png",
    "1e6da165-23cb-4426-91cf-450b3ac74e49.png",
    "86ae5f3b-1a9c-4895-87eb-b526b403f332.png",
    "4e53b13b-317a-484c-9b8b-7417453e45ff.png",
    "8ddb08c1-facb-4fb1-841b-bde4e47fe584.png",
    "41c3dbec-db54-4417-bb76-f51a4349943e.png",
    "a2881fcf-1af3-46e9-87b0-c198fad703d1.png",
    "808ec0fb-fcf8-4629-9296-38530521a95e.png",
    "7c411449-a3c6-48e4-ab30-d08bca5938d6.png",
    "0df28fee-6cef-48ae-9e8e-c0f0f3482412.png",
]

WALL_TEX_NAMES = [
    "plain copy 49.png",
    "18a6abbd-8ca1-43d9-9a0a-8424a7d1e016.png",
    "plain copy 2.png",
    "0882d081-d8be-49c0-aad6-3f5eff92265b.png",
    "plain copy 39.png",
    "7a19bf00-4f43-4e39-8fe6-977df7c6d147.png",
    "plain copy 8.png",
    "9b6f9830-a89d-4a42-965f-581dd0a12617.png",
    "bcf9bf9b-ac75-408e-ab2e-516391c39d8e.png",
    "plain copy 51.png",
    "845424a1-4603-47d7-892c-426c2d427f0d.png",
    "6e961263-5c90-41f9-87d3-b9e132c9583a.png",
    "c72ea705-e088-4665-9683-6a8685758aef.png",
    "plain copy 22.png",
    "plain copy 44.png",
    "plain copy 10.png",
    "df17f0f3-68bb-46e2-ae9f-86cf84f48b04.png",
    "plain copy 53.png",
    "0906d773-7c22-4173-9edc-efc6e2da88e0.png",
    "8995f82f-2435-4395-8ecd-a00824edd923.png",
    "0adf5feb-6a2f-4417-811a-b7d70d42fb34.png",
    "plain copy 5.png",
    "536410ee-0460-43b0-9e8e-799e96439a0d.png",
    "plain copy 11.png",
    "plain copy 19.png",
    "plain copy 17.png",
    "plain copy.png",
    "plain copy 31.png",
    "Lance_Z._white_square_tiles_texture_image_ikea_high_quality_rea_f41b5937-af3e-4df8-b19d-ae4fa942f741.png",
    "f6353155-1a10-41a1-be1d-149368938757.png",
    "b1a7c75f-6b03-4b10-99cd-3299d7b6beb5.png",
    "plain copy 52.png",
    "7844f4b6-7d34-46b4-8ea4-19ab954ca58f.png",
    "0dd40625-0cf8-4523-92cd-063c96108a23.png",
    "plain copy 40.png",
    "plain copy 21.png",
    "plain copy 36.png",
    "plain copy 37.png",
    "plain copy 41.png",
    "plain copy 23.png",
    "1b8b8a44-ce4e-4156-978b-ad66bf4a6f45.png",
    "plain copy 46.png",
    "d208d65f-4d99-4963-8140-d54180844c8c.png",
    "0aa2c331-8025-4417-a157-91467143c85b.png",
    "d4775c13-15ea-496d-b5fc-19d2a9cab128.png",
    "11760e4b-5f86-46cb-82ae-c87cc6f434d1.png",
    "plain copy 3.png",
    "plain copy 30.png",
    "plain copy 9.png",
    "0af727bd-14b4-4f7d-9fee-19f7f001ad34.png",
    "67d1229f-7689-4dd6-ab2b-d003e0cd3cb7.png",
    "plain copy 33.png",
    "plain copy 4.png",
    "2149f068-ca27-40ed-b53d-07e53210e678.png",
    "plain copy 38.png",
    "6b215002-fd31-4f1a-859a-d5d98a2a0c52.png",
    "4fe97e2e-ae88-4dc9-8321-3529a2108508.png",
    "Lance_Z._white_square_tiles_ikea_high_quality_realistic_ec4bd31e-d76d-4512-8b0d-a86dd8dd6aad.png",
    "plain copy 6.png",
    "plain copy 13.png",
    "0d7afefe-58c8-4161-bf56-0974c7e806a2.png",
    "plain copy 28.png",
    "314f4070-0b2f-4433-98e7-dbdf18a8bff2.png",
    "plain copy 18.png",
    "dae6863d-b653-4998-8239-c6a6bb24362c.png",
    "497cbc0b-a80a-42b3-8f5e-3bc97c145255.png",
    "plain copy 20.png",
    "4874a0d3-f5f2-46ea-a1d3-824b2f56f8ce.png",
    "plain copy 50.png",
    "plain copy 12.png",
    "7c7c9950-576a-45fd-b3d1-74cca30f6fe7.png",
    "8474c563-65b7-408b-976f-b19b38a44f90.png",
    "Lance_Z._white_square_tiles_texture_image_ikea_high_quality_rea_ce2dcd92-29d7-4176-b7fa-79d91dab133b.png",
    "738a1240-97f3-4fa6-97df-0ecc4b7aac26.png",
    "plain copy 32.png",
    "plain copy 29.png",
    "a4046f8f-a2d6-4ef4-aa39-2ec541cbd246.png",
    "plain copy 25.png",
    "cbf05cdc-5d1c-472b-9891-0e4802553359.png",
    "plain copy 16.png",
    "plain copy 47.png",
    "4bb0ad30-b60a-44a6-a259-058fab19b422.png",
    "77abb443-a1b8-4b02-96de-1a7bc68d4f3f.png",
    "plain copy 27.png",
    "plain copy 24.png",
    "plain copy 43.png",
    "35ded0fe-73a7-4e82-8c01-32f79f5214a3.png",
    "4b9d5219-38a0-4303-8837-29c4c51eed54.png",
    "8949ad7c-2b89-444c-a758-b513f25937e1.png",
    "plain copy 48.png",
    "plain copy 45.png",
    "plain copy 35.png",
    "plain copy 26.png",
    "plain copy 42.png",
    "f7c0d850-4b56-484f-85ce-e9c4bd01ec22.png",
    "9ec0b130-598a-4471-99ee-f68ee49a4a9f.png",
    "b1dcb5d2-434e-4bd1-9329-eb1c2a6c039d.png",
    "cf0d9763-3596-48de-af34-5f3647d812df.png",
    "plain copy 14.png",
    "fda88519-7a09-466d-84a2-19b216562dcb.png",
]


def get_random_textures(rng, frac=1.0):
    """
    This function returns a dictionary of random textures for cabinet, counter top, floor, and wall.

    Args:
        rng (np.random.Generator): Random number generator used for texture selection

        frac (float): Fraction of textures to select from the list of available textures
            Default: 1.0 (select from all textures)

    Returns:
        textures (dict): Dictionary of texture paths
    """

    end_ind = int(frac * 100)
    ind = rng.integers(0, end_ind)

    textures = dict(
        cab_tex=os.path.join(TEXTURES_DIR, "cabinet", CABINET_TEX_NAMES[ind]),
        counter_tex=os.path.join(TEXTURES_DIR, "counter", COUNTER_TOP_TEX_NAMES[ind]),
        floor_tex=os.path.join(TEXTURES_DIR, "floor", FLOOR_TEX_NAMES[ind]),
        wall_tex=os.path.join(TEXTURES_DIR, "wall", WALL_TEX_NAMES[ind]),
    )

    return textures


def replace_counter_top_texture(rng, initial_state: str, new_counter_top_texture_file: str = None):
    """
    This function replaces the counter top textures during playback.

    Args:
        rng (np.random.Generator): Random number generator used for texture selection

        initial_state (str): Initial env XML string

        new_counter_top_texture_file (str): New texture file for counter top: i.e "marble/dark_marble.png"
            If None (default), will replace with a random texture from marble directory
    """

    root = ET.fromstring(initial_state)
    asset = root.find("asset")

    if new_counter_top_texture_file is None:
        new_counter_top_texture_file = get_random_textures(rng)["counter_tex"]
    else:
        new_counter_top_texture_file = os.path.join(TEXTURES_DIR, new_counter_top_texture_file)

    # step 1: find the name of texture that will be replaced
    counter_tex_name = None
    for mat in asset.findall("material"):
        name = mat.get("name")
        if "counter_top" in name:
            counter_tex_name = mat.get("texture")
            break
    assert counter_tex_name is not None

    # step 2: find and replace texture element
    CTOP_TEX_NAME = "counter_top_replacement_texture"
    for tex in asset.findall("texture"):
        name = tex.get("name")
        if name == counter_tex_name:
            tex.set("name", CTOP_TEX_NAME)
            tex.set("file", str(new_counter_top_texture_file))

    # step 3: reference new textures in materials
    for mat in asset.findall("material"):
        name = mat.get("name")
        if "counter_top" in name:
            mat.set("texture", CTOP_TEX_NAME)

    return ET.tostring(root).decode("utf-8")


def replace_cab_textures(rng, initial_state: str, new_cab_texture_file: str = None):
    """
    This function replaces the cabinet and counter base textures during playback.

    Args:
        rng (np.random.Generator): Random number generator used for texture selection

        initial_state (str): Initial env XML string

        new_cab_texture_file (str): New texture file for counter base and cabinets: i.e "cabinet/..."
            If None (default), will replace with a random texture from flat or wood directories
    """

    root = ET.fromstring(initial_state)
    asset = root.find("asset")

    if new_cab_texture_file is None:
        new_cab_texture_file = get_random_textures(rng)["cab_tex"]
    else:
        new_cab_texture_file = os.path.join(TEXTURES_DIR, new_cab_texture_file)

    CAB_TEX_NAME_2D = "cab_replacement_texture_2d"
    tex_2d = find_elements(
        asset, tags="texture", attribs={"name": CAB_TEX_NAME_2D}, return_first=True
    )
    if tex_2d is not None:
        tex_2d.set("file", str(new_cab_texture_file))
    else:
        tex_2d = ET.Element(
            "texture", type="2d", name=CAB_TEX_NAME_2D, file=str(new_cab_texture_file)
        )
        asset.append(tex_2d)

    CAB_TEX_NAME_CUBE = "cab_replacement_texture_cube"
    tex_cube = find_elements(
        asset, tags="texture", attribs={"name": CAB_TEX_NAME_CUBE}, return_first=True
    )
    if tex_cube is not None:
        tex_cube.set("file", str(new_cab_texture_file))
    else:
        tex_cube = ET.Element(
            "texture",
            type="cube",
            name=CAB_TEX_NAME_CUBE,
            file=str(new_cab_texture_file),
        )
        asset.append(tex_cube)

    for mat in asset.findall("material"):
        name = mat.get("name")
        if "counter_base" in name:
            mat.set("texture", CAB_TEX_NAME_CUBE)
        elif "housing" in name:
            mat.set("texture", CAB_TEX_NAME_CUBE)
        elif (
            "stack" in name
            or "cab" in name
            or "shelves" in name
            or "bottom" in name
            or ("top" in name and "counter" not in name and "stove" not in name)
        ):
            if "handle" in name or "transparent" in name:
                continue
            elif "door" in name:
                mat.set("texture", CAB_TEX_NAME_2D)
            elif "shelves" in name:
                mat.set("texture", CAB_TEX_NAME_2D)
            else:
                mat.set("texture", CAB_TEX_NAME_CUBE)

    return ET.tostring(root).decode("utf-8")


def replace_floor_texture(rng, initial_state: str, new_floor_texture_file: str = None):
    """
    This function replaces the counter top textures during playback.

    Args:
        rng (np.random.Generator): Random number generator used for texture selection

        initial_state (str): Initial env XML string

        new_floor_texture_file (str): New texture file for counter top: i.e "wood/dark_wood_planks_2.png"
            If None (default), will replace with a random texture from wood directory
    """

    root = ET.fromstring(initial_state)
    asset = root.find("asset")

    if new_floor_texture_file is None:
        new_floor_texture_file = get_random_textures(rng)["floor_tex"]
    else:
        new_floor_texture_file = os.path.join(TEXTURES_DIR, new_floor_texture_file)

    # step 1: find the name of texture that will be replaced
    floor_tex_name = None
    for mat in asset.findall("material"):
        name = mat.get("name")
        if "floor" in name and "backing" not in name:
            floor_tex_name = mat.get("texture")
            break
    assert floor_tex_name is not None

    # step 2: find and replace texture element
    FLOOR_TEX_NAME = "floor_replacement_texture"
    for tex in asset.findall("texture"):
        name = tex.get("name")
        if name == floor_tex_name:
            tex.set("name", FLOOR_TEX_NAME)
            tex.set("file", str(new_floor_texture_file))
            tex.set("type", "2d")

    # step 3: reference new textures in materials
    for mat in asset.findall("material"):
        name = mat.get("name")
        if "floor" in name and "backing" not in name:
            mat.set("texture", FLOOR_TEX_NAME)
            mat.set("texrepeat", "2 2")

    return ET.tostring(root).decode("utf-8")


def replace_wall_texture(rng, initial_state: str, new_wall_texture_file: str = None):
    """
    This function replaces the counter top textures during playback.

    Args:
        rng (np.random.Generator): Random number generator used for texture selection

        initial_state (str): Initial env XML string

        new_wall_texture_file (str): New texture file for counter top: i.e "wood/dark_wood_planks_2.png"
            If None (default), will replace with a random texture from wood directory
    """

    root = ET.fromstring(initial_state)
    asset = root.find("asset")

    if new_wall_texture_file is None:
        new_wall_texture_file = get_random_textures(rng)["wall_tex"]
    else:
        new_wall_texture_file = os.path.join(TEXTURES_DIR, new_wall_texture_file)

    # step 1: find the name of texture that will be replaced
    wall_tex_name = None
    for mat in asset.findall("material"):
        name = mat.get("name")
        if "wall" in name and "floor" not in name and "backing" not in name:
            wall_tex_name = mat.get("texture")
            break
    assert wall_tex_name is not None

    # step 2: add new texture element
    WALL_TEX_NAME = "wall_replacement_texture"
    for tex in asset.findall("texture"):
        name = tex.get("name")
        if name == wall_tex_name:
            tex.set("name", WALL_TEX_NAME)
            tex.set("file", str(new_wall_texture_file))
            tex.set("type", "2d")

    # step 3: reference new textures in materials
    for mat in asset.findall("material"):
        name = mat.get("name")
        if "wall" in name and "floor" not in name and "backing" not in name:
            mat.set("texture", WALL_TEX_NAME)
            mat.set("texrepeat", "3 3")

    return ET.tostring(root).decode("utf-8")
