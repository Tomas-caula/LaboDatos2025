package aed;

class ArregloRedimensionableDeRecordatorios {


    public Recordatorio[] recordatorios;

    public ArregloRedimensionableDeRecordatorios() {
        // Implementar
        this.recordatorios = new Recordatorio[0];
    }

    public int longitud() {
        // Implementar
        return this.recordatorios.length;
    }

    public void agregarAtras(Recordatorio i) {
        // Implementar
        Recordatorio[] nuevoRecord = new Recordatorio[this.longitud() + 1];
        for (int j = 0 ; j < this.recordatorios.length; j++){
            nuevoRecord[j] = this.recordatorios[j];
        }
        nuevoRecord[nuevoRecord.length - 1] = i;
        this.recordatorios = nuevoRecord;
    }

    public Recordatorio obtener(int i) {
        // Implementar
        return this.recordatorios[i];
    }

    public void quitarAtras() {
        // Implementar
        Recordatorio[] nuevoRecord = new Recordatorio[this.longitud() - 1];
        for (int j = 0 ; j < this.recordatorios.length - 1; j++){
            nuevoRecord[j] = this.recordatorios[j];
        }
        this.recordatorios = nuevoRecord;
    }

    public void modificarPosicion(int indice, Recordatorio valor) {
        this.recordatorios[indice] = valor;
    }

    public ArregloRedimensionableDeRecordatorios(ArregloRedimensionableDeRecordatorios vector) {
        // Implementar
        this.recordatorios = new Recordatorio[0];
        for (int i = 0 ; i < vector.longitud(); i ++){
            this.agregarAtras(vector.obtener(i));
        }
    }

    public ArregloRedimensionableDeRecordatorios copiar() {
        // Implementar
        ArregloRedimensionableDeRecordatorios copia = new ArregloRedimensionableDeRecordatorios();
        for (int i = 0; i < this.longitud(); i++ ){
            copia.agregarAtras(this.obtener(i));
        }
        return copia;
    }

}   
