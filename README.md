# medieval_image_retrieval


## Configuración del Entorno

Para comenzar a trabajar con este proyecto, es necesario configurar el entorno de desarrollo. Este proyecto utiliza Conda como gestor de entornos y dependencias.

### Creación del Entorno Conda

Para crear el entorno Conda, ejecuta:

```bash
conda env create -f environment.yml
```

Esta instrucción creará un entorno Conda basado en la configuración especificada en `environment.yml`.

### Actualización del Entorno Conda

Si necesitas actualizar las dependencias y guardar los cambios en `environment.yml`, ejecuta:

```bash
conda env export > environment.yml
```

### Activación del Entorno

Una vez creado el entorno, actívalo con:

```bash
conda activate medieval_ret
```